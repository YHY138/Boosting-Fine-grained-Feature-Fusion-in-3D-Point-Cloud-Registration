"""REGTR network architecture
"""
import math

import torch
import torch.nn as nn

from backbone_kpconv.finegrained_kpconv import KPFEncoder, PreprocessorGPU, compute_overlaps
from generic_reg_model import GenericRegModel
from losses.corr_loss import CorrCriterion
from losses.feature_loss import InfoNCELossFull, CircleLossFull
from transformer.position_embedding import PositionEmbeddingCoordsSine, \
    PositionEmbeddingLearned, GeometricStructureEmbedding
from models.transformer.transformers import \
    TransformerCrossEncoderLayer, TransformerCrossEncoder

from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_inv, fast_compute_rigid_transform
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences
from utils.viz import visualize_registration
_TIMEIT = False

'''
model = Linformer(
        input_size=262144, # Dimension 1 of the input 数据个数，即有多少个点，类似这种
        channels=64, # Dimension 2 of the input 特征的描述信息。每个特征点对应的描述向量维度，类似于feature map的通道数
        dim_d=None, 直接用默认值就行# Overwrites the inner dim of the attention heads. If None, sticks with the recommended channels // nhead, as in the "Attention is all you need" paper
        dim_k=128, # 主要是这里是原REGTR里没有的。 k值越小，计算复杂度和时间越小，但性能会有所降低
        dim_ff=128, # Dimension in the feed forward network 这个应该和REGTR里面的d_feedforward变量
        dropout_ff=0.15, # Dropout for feed forward network 这里也可以直接用REGTR里面的dropout参数
        nhead=4, # Number of attention heads REGTR里面的nhead参数
        depth=2, # How many times to run the model ？？？深度，对这个模型执行几次？直接使用默认值1吧
        dropout=0.0, # How much dropout to apply to P_bar after softmax REGTR里面有这个参数
    # 下面的一些参数可以直接使用默认的就好
        activation="gelu", # What activation to use. Currently, only gelu and relu supported, and only on ff network.
        checkpoint_level="C0", # What checkpoint level to use. For more information, see below.
        parameter_sharing="layerwise", # What level of parameter sharing to use. For more information, see below.
        k_reduce_by_layer=0, # Going down `depth`, how much to reduce `dim_k` by, for the `E` and `F` matrices. Will have a minimum value of 1.
        full_attention=False, # Use full attention instead, for O(n^2) time and space complexity. Included here just for comparison
        include_ff=True, # Whether or not to include the Feed Forward layer
        w_o_intermediate_dim=None, # If not None, have 2 w_o matrices, such that instead of `dim*nead,channels`, you have `dim*nhead,w_o_int`, and `w_o_int,channels`
        ).cuda()
        
model = MHAttention(
        input_size=512, # Dimension 1 of the input
        channels=64, # Dimension 2 of the input
        dim=8, # Dim of each attn head
        dim_k=128, # What to sample the input length down to
        nhead=8, # Number of heads
        dropout=0, # Dropout for each of the heads
        activation="gelu", # Activation after attention has been concat'd
        checkpoint_level="C2", # If C2, checkpoint each of the heads
        parameter_sharing="layerwise", # What level of parameter sharing to do
        E_proj, F_proj, # The E and F projection matrices
        full_attention=False, # Use full attention instead
        w_o_intermediate_dim=None, # If not None, have 2 w_o matrices, such that instead of `dim*nead,channels`, you have `dim*nhead,w_o_int`, and `w_o_int,channels`
        )
        
E_proj = get_EF(input_size, dim_k, method, head_dim) # 通过这个函数来获得简单的E和F投射矩阵
'''


class RegTR(GenericRegModel):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!Using MY RegTR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(cfg.dataset)

        #######################
        # Preprocessor
        #######################
        self.preprocessor = PreprocessorGPU(cfg)

        #######################
        # KPConv Encoder/decoder
        #######################
        # print('#################### create KPConv')
        self.kpf_encoder = KPFEncoder(cfg, cfg.d_embed)
        # Bottleneck layer to shrink KPConv features to a smaller dimension for running attention
        self.feat_proj = nn.Linear(self.kpf_encoder.encoder_skip_dims[-1], cfg.d_embed, bias=True) # feat_proj层
        # print("#################### finish KPConv")

        #######################
        # Embeddings
        #######################
        self.Geoembeding = False
        if cfg.get('pos_emb_type', 'sine') == 'sine':
            self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.d_embed,
                                                         scale=cfg.get('pos_emb_scaling', 1.0))
        elif cfg['pos_emb_type'] == 'learned':
            self.pos_embed = PositionEmbeddingLearned(3, cfg.d_embed)
        elif cfg['pos_emb_type'] == 'GeoSin':
            print("!!!!!!!!!! 使用GeoSin位置编码")
            self.pos_embed = GeometricStructureEmbedding(cfg.d_embed)
            self.Geoembeding = True
        else:
            raise NotImplementedError
        # print('#################### finish poseembedding')

        #######################
        # Attention propagation
        #######################
        encoder_layer = TransformerCrossEncoderLayer(
            cfg.d_embed, cfg.nhead, cfg.d_feedforward, cfg.dropout,
            activation=cfg.transformer_act,
            normalize_before=cfg.pre_norm,
            sa_val_has_pos_emb=cfg.sa_val_has_pos_emb,
            ca_val_has_pos_emb=cfg.ca_val_has_pos_emb,
            attention_type=cfg.attention_type,
        )
        encoder_norm = nn.LayerNorm(cfg.d_embed) if cfg.pre_norm else None
        self.transformer_encoder = TransformerCrossEncoder(
            encoder_layer, cfg.num_encoder_layers, encoder_norm,
            return_intermediate=True)
        # print('#################### finish transencoder')

        #######################
        # Output layers
        #######################
        if cfg.get('direct_regress_coor', False):
            self.correspondence_decoder = CorrespondenceRegressor(cfg.d_embed)
        else:
            self.correspondence_decoder = CorrespondenceDecoder(cfg.d_embed,
                                                                cfg.corr_decoder_has_pos_emb,
                                                                self.pos_embed)
        # print('#################### finish regress corr')

        #######################
        # Losses
        #######################
        self.overlap_criterion = nn.BCEWithLogitsLoss()
        if self.cfg.feature_loss_type == 'infonce':
            self.feature_criterion = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
        elif self.cfg.feature_loss_type == 'circle':
            self.feature_criterion = CircleLossFull(dist_type='euclidean', r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = self.feature_criterion
        else:
            raise NotImplementedError

        self.corr_criterion = CorrCriterion(metric='mae')

        self.weight_dict = {}
        for k in ['overlap', 'feature', 'corr']:
            for i in cfg.get(f'{k}_loss_on', [cfg.num_encoder_layers - 1]):
                self.weight_dict[f'{k}_{i}'] = cfg.get(f'wt_{k}')
        self.weight_dict['feature_un'] = cfg.wt_feature_un

        self.logger.info('Loss weighting: {}'.format(self.weight_dict))
        self.logger.info(
            f'Config: d_embed:{cfg.d_embed}, nheads:{cfg.nhead}, pre_norm:{cfg.pre_norm}, '
            f'use_pos_emb:{cfg.transformer_encoder_has_pos_emb}, '
            f'sa_val_has_pos_emb:{cfg.sa_val_has_pos_emb}, '
            f'ca_val_has_pos_emb:{cfg.ca_val_has_pos_emb}'
        )

    def forward(self, batch):
        B = len(batch['src_xyz'])
        outputs = {}

        if _TIMEIT:
            t_start_all_cuda, t_end_all_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_pp_cuda, t_end_pp_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_all_cuda.record()
            t_start_pp_cuda.record()

        # Preprocess
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@  Current src and tgt point  @@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print(batch['src_xyz'][0].shape, batch['src_xyz'][1].shape, batch['tgt_xyz'][0].shape, batch['tgt_xyz'][1].shape)

        # 这里的preprocesser将输入的src和tgt点云做了一次处理，使得点的总数 变了？？
        kpconv_meta = self.preprocessor(batch['src_xyz'] + batch['tgt_xyz']) # 括号内相加的结果为：[src三维点数组, tgt三维点数组]
        batch['kpconv_meta'] = kpconv_meta
        slens = [s.tolist() for s in kpconv_meta['stack_lengths']]
        slens_c = slens[-1]
        src_slens_c, tgt_slens_c = slens_c[:B], slens_c[B:]
        feats0 = torch.ones_like(kpconv_meta['points'][0][:, 0:1])

        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@  source data of KPConv  @@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print(kpconv_meta['points'][0].shape)

        if _TIMEIT:
            t_end_pp_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_pp_cuda = t_start_pp_cuda.elapsed_time(t_end_pp_cuda) / 1000
            t_start_enc_cuda, t_end_enc_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_enc_cuda.record()

        ####################
        # REGTR Encoder
        ####################
        # KPConv encoder (downsampling) to obtain unconditioned features
        # 使用骨干网络卷积特征描述向量feats0。kpconv_meta是点的三维坐标？
        # print('kpconv_meta data is {}'.format(kpconv_meta.keys()))
        feats_un, skip_x = self.kpf_encoder(feats0, kpconv_meta)
        # print('dddddddddddddddddddddd!!!!!!!!!!!!! after kpconv features un shape !!!!!!!!!!!!!!!!!')
        # print(feats_un.shape)

        if _TIMEIT:
            t_end_enc_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_enc_cuda = t_start_enc_cuda.elapsed_time(t_end_enc_cuda) / 1000
            t_start_att_cuda, t_end_att_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_att_cuda.record()

        both_feats_un = self.feat_proj(feats_un)
        src_feats_un, tgt_feats_un = split_src_tgt(both_feats_un, slens_c) # 训练的时候batch等于2，所以src_feats_un包含了两个训练样本

        # Position embedding for downsampled points
        src_xyz_c, tgt_xyz_c = split_src_tgt(kpconv_meta['points'][-1], slens_c) # 这里的src是特征点的三维坐标
        # print('###################src_xyz_c shape is {}, src_feats_un shape is {}'.format(src_xyz_c[-1].shape, src_feats_un[-1].shape))

        if self.Geoembeding:
            # print('!@@@@@@@@@@@@ ', kpconv_meta['points'][-1])
            # print("!@@@@@@@@@@ ", src_xyz_c[0].unsqueeze(0))
            src_pe, tgt_pe = [], []
            for xyzi in range(len(src_xyz_c)):
                src_pe.append(self.pos_embed(src_xyz_c[xyzi].unsqueeze(0)))
                tgt_pe.append(self.pos_embed(tgt_xyz_c[xyzi].unsqueeze(0)))
            src_pe = tuple(src_pe)
            tgt_pe = tuple(tgt_pe)
            # print("############ ",type(src_pe), "   ", src_pe[0].shape, "  ", src_pe[1].shape)
        else:
            # 根据Source和Target三维点坐标，进行位置编码
            src_pe, tgt_pe = split_src_tgt(self.pos_embed(kpconv_meta['points'][-1]), slens_c)
            # print("############ ",type(src_pe), "   ", src_pe[0].shape, "  ", src_pe[1].shape)
            # 正常的src-pe的格式为：（tensor1, tensor2）元组的形式,tensorsize==（N，512），src-pe的格式为：
        src_pe_padded, _, _ = pad_sequence(src_pe)
        tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

        # Performs padding, then apply attention (REGTR "encoder" stage) to condition on the other
        # point cloud
        src_feats_padded, src_key_padding_mask, _ = pad_sequence(src_feats_un,
                                                                 require_padding_mask=True)
        tgt_feats_padded, tgt_key_padding_mask, _ = pad_sequence(tgt_feats_un,
                                                                 require_padding_mask=True)
        # print('!!!!!!!!!!!!!!!!!!!!!!!transformer encoder src_feats_cond input!!!!!!!!!!')
        # print(src_feats_padded[-1])
        # print( src_xyz_c[-1].shape, src_xyz_c[0].shape)
        src_feats_cond, tgt_feats_cond = self.transformer_encoder(
            src_feats_padded, tgt_feats_padded,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_pos=src_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
            tgt_pos=tgt_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
            # src_xyz=src_xyz_c,
            # tgt_xyz=tgt_xyz_c,
        )
        # print(src_feats_cond.shape)
        # print('!!!!!!!!!!!!!!!!!!!!!!!After transformer encoder srcfeats_cond output!!!!!!!!!!!')

        src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list = \
            self.correspondence_decoder(src_feats_cond, tgt_feats_cond, src_xyz_c, tgt_xyz_c)

        src_feats_list = unpad_sequences(src_feats_cond, src_slens_c)
        tgt_feats_list = unpad_sequences(tgt_feats_cond, tgt_slens_c)
        num_pred = src_feats_cond.shape[0]

        ## TIMING CODE
        if _TIMEIT:
            t_end_att_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_att_cuda = t_start_att_cuda.elapsed_time(t_end_att_cuda) / 1000
            t_start_pose_cuda, t_end_pose_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_pose_cuda.record()

        # Stacks correspondences in both directions and computes the pose
        corr_all, overlap_prob = [], []
        for b in range(B):
            corr_all.append(torch.cat([
                torch.cat([src_xyz_c[b].expand(num_pred, -1, -1), src_corr_list[b]], dim=2),
                torch.cat([tgt_corr_list[b], tgt_xyz_c[b].expand(num_pred, -1, -1)], dim=2)
            ], dim=1))
            overlap_prob.append(torch.cat([
                torch.sigmoid(src_overlap_list[b][:, :, 0]),
                torch.sigmoid(tgt_overlap_list[b][:, :, 0]),
            ], dim=1))

        # pred_pose_weighted = torch.stack([
        #     compute_rigid_transform(corr_all[b][..., :3], corr_all[b][..., 3:],
        #                             overlap_prob[b])
        #     for b in range(B)], dim=1)
        # The fast function can be either enabled during training or disabled during training and activated during testing. 
        # Activating it during testing enhances the computational efficiency of the pose estimation module without compromising accuracy.
        pred_pose_weighted = torch.stack([
            fast_compute_rigid_transform(corr_all[b][..., :3], corr_all[b][..., 3:],
                                    overlap_prob[b])
            for b in range(B)], dim=1)

        ## TIMING CODE
        if _TIMEIT:
            t_end_pose_cuda.record()
            t_end_all_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_pose_cuda = t_start_pose_cuda.elapsed_time(t_end_pose_cuda) / 1000
            t_elapsed_all_cuda = t_start_all_cuda.elapsed_time(t_end_all_cuda) / 1000
            with open('timings.txt', 'a') as fid:
                fid.write('{:10f}\t{:10f}\t{:10f}\t{:10f}\t{:10f}\n'.format(
                    t_elapsed_pp_cuda, t_elapsed_enc_cuda, t_elapsed_att_cuda,
                    t_elapsed_pose_cuda, t_elapsed_all_cuda
                ))

        outputs = {
            # Predictions
            'src_feat_un': src_feats_un,
            'tgt_feat_un': tgt_feats_un,
            'src_feat': src_feats_list,  # List(B) of (N_pred, N_src, D)
            'tgt_feat': tgt_feats_list,  # List(B) of (N_pred, N_tgt, D)

            'src_kp': src_xyz_c,
            'src_kp_warped': src_corr_list,
            'tgt_kp': tgt_xyz_c,
            'tgt_kp_warped': tgt_corr_list,

            'src_overlap': src_overlap_list,
            'tgt_overlap': tgt_overlap_list,

            'pose': pred_pose_weighted,
        }
        return outputs

    def compute_loss(self, pred, batch):
        losses = {}
        kpconv_meta = batch['kpconv_meta']
        pose_gt = batch['pose']
        p = len(kpconv_meta['stack_lengths']) - 1  # coarsest level

        # Compute groundtruth overlaps first
        batch['overlap_pyr'] = compute_overlaps(batch)
        src_overlap_p, tgt_overlap_p = \
            split_src_tgt(batch['overlap_pyr'][f'pyr_{p}'], kpconv_meta['stack_lengths'][p])

        # Overlap prediction loss
        all_overlap_pred = torch.cat(pred['src_overlap'] + pred['tgt_overlap'], dim=-2)
        all_overlap_gt = batch['overlap_pyr'][f'pyr_{p}']
        for i in self.cfg.overlap_loss_on:
            losses[f'overlap_{i}'] = self.overlap_criterion(all_overlap_pred[i, :, 0], all_overlap_gt)

        # Feature criterion
        for i in self.cfg.feature_loss_on:
            losses[f'feature_{i}'] = self.feature_criterion(
                [s[i] for s in pred['src_feat']],
                [t[i] for t in pred['tgt_feat']],
                se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
            )
        losses['feature_un'] = self.feature_criterion_un(
            pred['src_feat_un'],
            pred['tgt_feat_un'],
            se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
        )

        # Loss on the 6D correspondences
        for i in self.cfg.corr_loss_on:
            src_corr_loss = self.corr_criterion(
                pred['src_kp'],
                [w[i] for w in pred['src_kp_warped']],
                batch['pose'],
                overlap_weights=src_overlap_p
            )
            tgt_corr_loss = self.corr_criterion(
                pred['tgt_kp'],
                [w[i] for w in pred['tgt_kp_warped']],
                torch.stack([se3_inv(p) for p in batch['pose']]),
                overlap_weights=tgt_overlap_p
            )
            losses[f'corr_{i}'] = src_corr_loss + tgt_corr_loss

        debug = False  # Set this to true to look at the registration result
        if debug:
            b = 0
            o = -1  # Visualize output of final transformer layer
            visualize_registration(batch['src_xyz'][b], batch['tgt_xyz'][b],
                                   torch.cat([pred['src_kp'][b], pred['src_kp_warped'][b][o]], dim=1),
                                   correspondence_conf=torch.sigmoid(pred['src_overlap'][b][o])[:, 0],
                                   pose_gt=pose_gt[b], pose_pred=pred['pose'][o, b])

        losses['total'] = torch.sum(
            torch.stack([(losses[k] * self.weight_dict[k]) for k in losses]))
        return losses


class CorrespondenceDecoder(nn.Module):
    def __init__(self, d_embed, use_pos_emb, pos_embed=None, num_neighbors=0):
        super().__init__()

        assert use_pos_emb is False or pos_embed is not None, \
            'Position encoder must be supplied if use_pos_emb is True'

        self.use_pos_emb = use_pos_emb
        self.pos_embed = pos_embed
        self.q_norm = nn.LayerNorm(d_embed)

        self.q_proj = nn.Linear(d_embed, d_embed)
        self.k_proj = nn.Linear(d_embed, d_embed)
        self.conf_logits_decoder = nn.Linear(d_embed, 1)
        self.num_neighbors = num_neighbors

        # nn.init.xavier_uniform_(self.q_proj.weight)
        # nn.init.xavier_uniform_(self.k_proj.weight)

    def simple_attention(self, query, key, value, key_padding_mask=None):
        """Simplified single-head attention that does not project the value:
        Linearly projects only the query and key, compute softmax dot product
        attention, then returns the weighted sum of the values

        Args:
            query: ([N_pred,] Q, B, D)
            key: ([N_pred,] S, B, D)
            value: (S, B, E), i.e. dimensionality can be different
            key_padding_mask: (B, S)

        Returns:
            Weighted values (B, Q, E)
        """

        q = self.q_proj(query) / math.sqrt(query.shape[-1])
        k = self.k_proj(key)

        attn = torch.einsum('...qbd,...sbd->...bqs', q, k)  # (B, N_query, N_src)

        if key_padding_mask is not None:
            attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(key_padding_mask, float('-inf'))
            attn = attn + attn_mask[:, None, :]  # ([N_pred,] B, Q, S)

        if self.num_neighbors > 0:
            neighbor_mask = torch.full_like(attn, fill_value=float('-inf'))
            haha = torch.topk(attn, k=self.num_neighbors, dim=-1).indices
            neighbor_mask[:, :, haha] = 0
            attn = attn + neighbor_mask

        attn = torch.softmax(attn, dim=-1)

        attn_out = torch.einsum('...bqs,...sbd->...qbd', attn, value)

        return attn_out

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3)
            tgt_xyz: List of ([N_pred,] N_tgt, 3)

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)
        assert src_xyz_padded.shape[:-1] == src_feats_padded.shape[-3:-1] and \
               tgt_xyz_padded.shape[:-1] == tgt_feats_padded.shape[-3:-1]

        if self.use_pos_emb:
            both_xyz_packed = torch.cat(src_xyz + tgt_xyz)
            slens = list(map(len, src_xyz)) + list(map(len, tgt_xyz))
            src_pe, tgt_pe = split_src_tgt(self.pos_embed(both_xyz_packed), slens)
            src_pe_padded, _, _ = pad_sequence(src_pe)
            tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

        # Decode the coordinates
        src_feats2 = src_feats_padded + src_pe_padded if self.use_pos_emb else src_feats_padded
        tgt_feats2 = tgt_feats_padded + tgt_pe_padded if self.use_pos_emb else tgt_feats_padded
        src_corr = self.simple_attention(src_feats2, tgt_feats2, pad_sequence(tgt_xyz)[0],
                                         tgt_key_padding_mask)
        tgt_corr = self.simple_attention(tgt_feats2, src_feats2, pad_sequence(src_xyz)[0],
                                         src_key_padding_mask)

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list


class CorrespondenceRegressor(nn.Module):

    def __init__(self, d_embed):
        super().__init__()

        self.coor_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 3)
        )
        self.conf_logits_decoder = nn.Linear(d_embed, 1)

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3). Ignored
            tgt_xyz: List of ([N_pred,] N_tgt, 3). Ignored

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)

        # Decode the coordinates
        src_corr = self.coor_mlp(src_feats_padded)
        tgt_corr = self.coor_mlp(tgt_feats_padded)

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list
