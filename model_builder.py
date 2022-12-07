import torch
import torch.nn as nn
import torch.nn.functional as F

class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()
        # encoder
        self.f = nn.Sequential(
            nn.Linear(input_dimension, input_dimension),
            nn.BatchNorm1d(input_dimension, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(input_dimension, input_dimension),
            nn.BatchNorm1d(input_dimension, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(input_dimension, output_dimension),
        )

    def forward(self, x):
        x = self.f(x)
        return x


class Visual_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Visual_Embedding_Unit, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dimension, 512),
            nn.BatchNorm1d(512, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, output_dimension),
        )
    def forward(self, x):
        x = self.fc(x)
        return x


class Fused_Gated_Unit(nn.Module):
    def __init__(self, input_dimension1, input_dimension2, output_dimension):
        super(Fused_Gated_Unit, self).__init__()

        self.fc_img = nn.Linear(input_dimension1, output_dimension)
        self.fc_text = nn.Linear(input_dimension2, output_dimension)
        self.fc_out = nn.Linear(output_dimension + output_dimension, output_dimension)
        self.BN = nn.BatchNorm1d(output_dimension, track_running_stats=True)
        self.cg = Context_Gating(output_dimension)

    def forward(self, img, text):
        img = self.fc_img(img)
        text = self.fc_text(text)
        x = torch.cat((img, text), 1)
        x = self.fc_out(x)
        x = self.BN(x)
        x = self.cg(x)
        return x


class Context_Gating(nn.Module):
    def __init__(self, dimension):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension, bias=True)

    def forward(self, x):
        x1 = self.fc(x)
        x = torch.cat((x, x1), 1)
        return F.glu(x, 1)


class Sentence_Maxpool(nn.Module):
    def __init__(self, word_dimension, output_dim):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim, bias=True)
        self.BN = nn.BatchNorm1d(output_dim, track_running_stats=True)
    def forward(self, x):
        x = self.fc(x)
        x = self.BN(x)
        return F.leaky_relu(x)


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class Net(nn.Module):

    def __init__(self, dim=512, K=65536, m=0.999, T=0.07,
            embd_dim = 1024,
            video_dim = 2048,
            img_dim = 2048,
            we_dim = 768):
        """
        dim: feature dimension (default: 512)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(Net, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dim = dim

        m1 = [0,1,2,3,4]
        m2 = [5,6,7]
        m3 = [9,10,11,12,13]
        m4 = [14,15,16,17]
        m5 = [19,20,21,22]
        m6 = [23,24,25,26,27,28,29]
        self.labelmap = [m1,m2,m3,m4,m5,m6]

        self.video_cluster_mean1 = torch.randn(dim, 6).cuda()
        self.video_cluster_mean2 = torch.randn(dim, 30).cuda()
        self.product_cluster_mean1 = torch.randn(dim, 6).cuda()
        self.product_cluster_mean2 = torch.randn(dim, 30).cuda()

        self.product_similarity_matrix1 = torch.randn(6, 6).cuda()
        self.product_similarity_matrix2 = torch.randn(30, 30).cuda()
        self.video_similarity_matrix1 = torch.randn(6, 6).cuda()
        self.video_similarity_matrix2 = torch.randn(30, 30).cuda()

        # create the encoders
        self.GU_q_video = Visual_Embedding_Unit(video_dim, embd_dim)
        self.GU_q_img = Visual_Embedding_Unit(img_dim, embd_dim)
        self.GU_q_vtext = Sentence_Maxpool(we_dim, embd_dim)
        self.GU_q_ptext = Sentence_Maxpool(we_dim, embd_dim)
        self.GU_q_video_fuse = Fused_Gated_Unit(embd_dim,embd_dim, embd_dim)
        self.GU_q_product_fuse = Fused_Gated_Unit(embd_dim ,embd_dim, embd_dim)

        self.GU_k_video = Visual_Embedding_Unit(video_dim, embd_dim)
        self.GU_k_img = Visual_Embedding_Unit(img_dim, embd_dim)
        self.GU_k_vtext = Sentence_Maxpool(we_dim, embd_dim)
        self.GU_k_ptext = Sentence_Maxpool(we_dim, embd_dim)
        self.GU_k_video_fuse = Fused_Gated_Unit(embd_dim, embd_dim, embd_dim)
        self.GU_k_product_fuse = Fused_Gated_Unit(embd_dim, embd_dim, embd_dim)

        self.encoder_q_video = Gated_Embedding_Unit(embd_dim, dim)
        self.encoder_k_video = Gated_Embedding_Unit(embd_dim, dim)
        self.encoder_q_product = Gated_Embedding_Unit(embd_dim, dim)
        self.encoder_k_product = Gated_Embedding_Unit(embd_dim, dim)

        # create the queue
        self.register_buffer("queue", torch.randn(dim,K,30))
        self.queue_video = nn.functional.normalize(self.queue, dim=0).cuda()
        self.queue_product = nn.functional.normalize(self.queue, dim=0).cuda()

        self.register_buffer("queue_cluster1", torch.zeros(6, K, 30))
        self.register_buffer("queue_cluster2", torch.zeros(30, K, 30))
        self.register_buffer("queue_cluster3", torch.zeros(320, K, 30))
        self.queue_video_cluster1 = nn.functional.normalize(self.queue_cluster1, dim=0).cuda()
        self.queue_video_cluster2 = nn.functional.normalize(self.queue_cluster2, dim=0).cuda()
        self.queue_video_cluster3 = nn.functional.normalize(self.queue_cluster3, dim=0).cuda()
        self.queue_product_cluster1 = nn.functional.normalize(self.queue_cluster1, dim=0).cuda()
        self.queue_product_cluster2 = nn.functional.normalize(self.queue_cluster2, dim=0).cuda()
        self.queue_product_cluster3 = nn.functional.normalize(self.queue_cluster3, dim=0).cuda()

        self.register_buffer("queue_ptr", torch.zeros(30, dtype=torch.long))

        self.model_pairs = [
                            [self.encoder_q_video, self.encoder_k_video],
                            [self.encoder_q_product, self.encoder_k_product],
                            ]
        self.copy_params()


    # utils
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
            Momentum update of the key encoder
        """
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.m + param.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, str, label):
        label = int(label)
        #batch_size = keys.shape[0]
        keys = keys.unsqueeze(0)
        batch_size = 1
        ptr = int(self.queue_ptr[label])
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        if(str == "video"):
            self.queue_video[:, ptr:ptr + batch_size, label] = keys.T
        if(str == "first_video_cluster"):
            self.queue_video_cluster1[:, ptr:ptr + batch_size, label] = keys.T
        if (str == "second_video_cluster"):
            self.queue_video_cluster2[:, ptr:ptr + batch_size, label] = keys.T
        if (str == "third_video_cluster"):
            self.queue_video_cluster3[:, ptr:ptr + batch_size, label] = keys.T

        if(str == "product"):
            self.queue_product[:, ptr:ptr + batch_size,label] = keys.T
        if(str == "first_product_cluster"):
            self.queue_product_cluster1[:, ptr:ptr + batch_size, label] = keys.T
        if (str == "second_product_cluster"):
            self.queue_product_cluster2[:, ptr:ptr + batch_size, label] = keys.T
        if (str == "third_product_cluster"):
            self.queue_product_cluster3[:, ptr:ptr + batch_size, label] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[label] = ptr

    @torch.no_grad()
    def _update_cluster_mean_(self):
        for label in range(0, 30): # update the center of middle ontology
            feature = self.queue_video[:, :, label]
            feature = feature.squeeze(1)
            feature = torch.mean(feature, dim=1)
            feature = torch.where(torch.isnan(feature), torch.full_like(feature, 1), feature)
            self.video_cluster_mean2[:, label] = feature

            feature = self.queue_product[:, :, label]
            feature = feature.squeeze(1)
            feature = torch.mean(feature, dim=1)
            feature = torch.where(torch.isnan(feature), torch.full_like(feature, 1), feature)
            self.product_cluster_mean2[:, label] = feature

        for label in range(0, 6): # update the center of upper ontology
            feature = self.video_cluster_mean2[:, int(self.labelmap[label][0])].unsqueeze(1)
            for i in range(1, len(self.labelmap[label])):
                tp = self.video_cluster_mean2[:, int(self.labelmap[label][i])].unsqueeze(1)
                feature = torch.cat([feature, tp], dim=1)
            feature = torch.mean(feature, dim=1)
            feature = torch.where(torch.isnan(feature), torch.full_like(feature, 1), feature)
            self.video_cluster_mean1[:, label] = feature

            feature = self.product_cluster_mean2[:, int(self.labelmap[label][0])].unsqueeze(1)
            for i in range(1, len(self.labelmap[label])):
                tp = self.product_cluster_mean2[:, int(self.labelmap[label][i])].unsqueeze(1)
                feature = torch.cat([feature, tp], dim=1)
            feature = torch.mean(feature, dim=1)
            feature = torch.where(torch.isnan(feature), torch.full_like(feature, 1), feature)
            self.product_cluster_mean1[:, label] = feature

        for i in range(0, 6):
            for j in range(0, 6):
                self.product_similarity_matrix1[i][j] = torch.pairwise_distance(self.product_cluster_mean1[:, i].unsqueeze(0),
                                                                            self.product_cluster_mean1[:, j].unsqueeze(0), p=2).cuda()
                self.video_similarity_matrix1[i][j] = torch.pairwise_distance(self.video_cluster_mean1[:, i].unsqueeze(0),
                                                                          self.video_cluster_mean1[:, j].unsqueeze(0), p=2).cuda()
        for i in range(0, 30):
            for j in range(0, 30):
                self.product_similarity_matrix2[i][j] = torch.pairwise_distance(self.product_cluster_mean2[:, i].unsqueeze(0),
                                                                            self.product_cluster_mean2[:, j].unsqueeze(0), p=2).cuda()
                self.video_similarity_matrix2[i][j] = torch.pairwise_distance(self.video_cluster_mean2[:, i].unsqueeze(0),
                                                                          self.video_cluster_mean2[:, j].unsqueeze(0), p=2).cuda()

    @torch.no_grad()
    def _get_cluster_mean_(self, label, v2p=0, level=1):
        #label = int(label.item())
        if(v2p==0):
            if(level==1):
                return self.video_cluster_mean1[:, label.view(1).long()]
            else:
                return self.video_cluster_mean2[:, label.view(1).long()]
        else:
            if(level==1):
                return self.product_cluster_mean1[:, label.view(1).long()]
            else:
                return self.product_cluster_mean2[:, label.view(1).long()]

    @torch.no_grad()
    def cacu_weight(self, first_cluster, second_cluster, third_cluster, klabel1, klabel2, klabel3, v2p):
        k_cluster1 = klabel1[:, :, 0].squeeze(0)
        for i in range(1, 30):
            kp = klabel1[:, :, i].squeeze(0)
            k_cluster1 = torch.cat((k_cluster1, kp), dim=1)

        k_cluster2 = klabel2[:, :, 0].squeeze(0)
        for i in range(1, 30):
            kp = klabel2[:, :, i].squeeze(0)
            k_cluster2 = torch.cat((k_cluster2, kp), dim=1)

        k_cluster3 = klabel3[:, :, 0].squeeze(0)
        for i in range(1, 30):
            kp = klabel3[:, :, i].squeeze(0)
            k_cluster3 = torch.cat((k_cluster3, kp), dim=1)

        a1 = torch.mm(first_cluster, k_cluster1, out=None).cuda()
        a2 = torch.mm(second_cluster, k_cluster2, out=None).cuda()
        a3 = torch.mm(third_cluster, k_cluster3, out=None).cuda()
        x = torch.full([a1.shape[0], a1.shape[1]], 1.0).cuda()

        if(v2p == 1):
            # Micro-video -> Product: upper category ontology contrast
            y1 = torch.matmul(first_cluster, self.product_similarity_matrix1)
            y1 = torch.matmul(y1, k_cluster1)
            y1 = torch.exp(y1)
            # Micro-video -> Product: middle category ontology contrast
            y2 = torch.matmul(second_cluster, self.product_similarity_matrix2)
            y2 = torch.matmul(y2, k_cluster2)
            y2 = torch.exp(y2)

        else:
            # Product -> Micro-video: upper category ontology contrast
            y1 = torch.matmul(first_cluster, self.video_similarity_matrix1)
            y1 = torch.matmul(y1, k_cluster1)
            y1 = torch.exp(y1)
            # Product -> Micro-video: middle category ontology contrast
            y2 = torch.matmul(second_cluster, self.video_similarity_matrix2)
            y2 = torch.matmul(y2, k_cluster2)
            y2 = torch.exp(y2)


        y1 = F.normalize(y1).cuda()
        y2 = F.normalize(y2).cuda()

        w1 = x - y1 - y2
        w2 = x - y2

        weight = torch.where(a1 > 0, x, w1).cuda()
        weight = torch.where(a2 > 0, weight, w2).cuda()

        return weight

    def forward(self, video, img, videotext, imgtext, queue_index, first_cluster, second_cluster, third_cluster,args, tag='train'):

        videoq = self.GU_q_video(video)
        imgq = self.GU_q_img(img)
        vtextq = self.GU_q_vtext(videotext)
        ptextq = self.GU_q_ptext(imgtext)
        im_q_video = self.GU_q_video_fuse(videoq, vtextq)
        im_q_product = self.GU_q_product_fuse(imgq, ptextq)


        videok = self.GU_q_video(video)
        imgk = self.GU_q_img(img)
        vtextk = self.GU_q_vtext(videotext)
        ptextk = self.GU_q_ptext(imgtext)
        im_k_video = self.GU_q_video_fuse(videok, vtextk)
        im_k_product = self.GU_q_product_fuse(imgk, ptextk)

        if(tag == 'test'):
            vq = self.encoder_q_video(im_q_video)
            pq = self.encoder_q_product(im_q_product)
            vk = self.encoder_k_video(im_k_video)
            pk = self.encoder_k_product(im_k_product)

            return vq, pq, vk, pk

        # compute query features
        q_video = self.encoder_q_video(im_q_video)  # queries: NxC
        q_video = nn.functional.normalize(q_video, dim=1)

        q_product = self.encoder_q_product(im_q_product)
        q_product = nn.functional.normalize(q_product, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k_video = self.encoder_k_video(im_k_video)  # keys: NxC
            k_video = nn.functional.normalize(k_video, dim=1)

            k_product = self.encoder_k_product(im_k_product)  # keys: NxC
            k_product = nn.functional.normalize(k_product, dim=1)

        q_video = q_video.squeeze(1)
        k_video = k_video.squeeze(1)


        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_video, k_product]).unsqueeze(-1)
        # negative logits: NxK
        pq = self.queue_product.clone().detach().cuda()
        pqq = pq[:, :, 0].squeeze(1)
        for i in range(1, 30):
            pq1 = pq[:, :, i].squeeze(1)
            pqq = torch.cat((pqq, pq1), 1)

        l_neg = torch.einsum('nc,ck->nk', [q_video, pqq.cuda()])


        #calculate the weight matrix of negative queue
        klabel1 = self.queue_product_cluster1.clone().detach().cuda()
        klabel2 = self.queue_product_cluster2.clone().detach().cuda()
        klabel3 = self.queue_product_cluster3.clone().detach().cuda()


        weight = self.cacu_weight(first_cluster, second_cluster, third_cluster, klabel1, klabel2, klabel3, v2p=1).cuda()
        # The negative logits is multiplied by the weight matrix
        l_neg = l_neg.mul(weight).cuda()

        # logits: Nx(1+K)
        logits1 = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits1 /= self.T

        # labels: positive key indicators
        labels1 = torch.zeros(logits1.shape[0], dtype=torch.long).cuda()

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_product, k_video]).unsqueeze(-1)
        # negative logits: NxK
        vq = self.queue_video.clone().detach().cuda()
        vqq = vq[:, :, 0].squeeze(1)
        for i in range(1, 30):
            vq1 = vq[:, :, i].squeeze(1)
            vqq = torch.cat((vqq, vq1), 1)

        l_neg = torch.einsum('nc,ck->nk', [q_product, vqq.cuda()])

        klabel1 = self.queue_video_cluster1.clone().detach().cuda()
        klabel2 = self.queue_video_cluster2.clone().detach().cuda()
        klabel3 = self.queue_video_cluster3.clone().detach().cuda()

        weight = self.cacu_weight(first_cluster, second_cluster, third_cluster, klabel1, klabel2, klabel3, v2p=0).cuda()
        # The negative logits is multiplied by the weight matrix
        l_neg = l_neg.mul(weight).cuda()

        # logits: Nx(1+K)
        logits2 = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits2 /= self.T

        # labels: positive key indicators
        labels2 = torch.zeros(logits2.shape[0], dtype=torch.long).cuda()


        # dequeue and enqueue
        for inn in range(0, args.batch_size):
            self._dequeue_and_enqueue(k_video[inn,:], "video", queue_index[inn])
            self._dequeue_and_enqueue(first_cluster[inn,:], "first_video_cluster", queue_index[inn])
            self._dequeue_and_enqueue(second_cluster[inn,:], "second_video_cluster", queue_index[inn])
            self._dequeue_and_enqueue(third_cluster[inn,:], "third_video_cluster", queue_index[inn])
            self._dequeue_and_enqueue(k_product[inn,:], "product", queue_index[inn])
            self._dequeue_and_enqueue(first_cluster[inn,:], "first_product_cluster", queue_index[inn])
            self._dequeue_and_enqueue(second_cluster[inn,:], "second_product_cluster", queue_index[inn])
            self._dequeue_and_enqueue(third_cluster[inn,:], "third_product_cluster", queue_index[inn])
            #update
            self._update_cluster_mean_()

        return logits1, labels1, logits2, labels2, videoq, imgq, vtextq, ptextq

