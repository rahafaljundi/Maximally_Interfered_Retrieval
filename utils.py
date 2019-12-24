import torch
import torch.nn.functional as F
import numpy as np
import copy
import pdb
from collections import OrderedDict as OD
from collections import defaultdict as DD

torch.random.manual_seed(0)

''' For MIR '''
def overwrite_grad(pp, new_grad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        param.grad=torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(
            param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1
def add_grad(pp, new_grad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        #param.grad=torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(
            param.data.size())

        param.grad.data.add_(this_grad)
        cnt += 1


def get_grad_vector(args, pp, grad_dims):
    """
     gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims))
    if args.cuda: grads = grads.cuda()

    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads

def get_future_step_parameters(this_net,grad_vector,grad_dims,lr=1):
    """
    computes \theta-\delta\theta
    :param this_net:
    :param grad_vector:
    :return:
    """
    new_net=copy.deepcopy(this_net)
    overwrite_grad(new_net.parameters,grad_vector,grad_dims)
    with torch.no_grad():
        for param in new_net.parameters():
            if param.grad is not None:

                param.data=param.data - lr*param.grad.data
    return new_net

def optimize(args,model,opt,input_x,input_y,mem_x,mem_y,far_mem_x,task):


    if args.mask:  # MASK OUT UN Replayed or seen classes
        yy = torch.cat((input_y, mem_y), dim=0)

        mask = get_mask_unused_memories(yy, args.n_classes)
    inner_iter=1
    while True:

        logits=model(input_x)
        logits_buffer = model(mem_x)
        if args.mask:
            logits = logits.masked_fill(mask, 1e-9)
            logits_buffer = logits_buffer.masked_fill(mask, 1e-9)

        loss = F.cross_entropy(logits, input_y)
        loss.backward()  # Loss of the new batch
        b_loss=(args.multiplier * F.cross_entropy(logits_buffer, mem_y))
        b_loss.backward()#Loss of the neighbours
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        model_temp = get_future_step_parameters_with_grads(model,  args.lr)

        KL=- args.kl_far *(F.softmax(model(far_mem_x).detach(), dim=1) * F.log_softmax(model_temp(far_mem_x), dim=1)).sum(dim=1).mean()
        
        KL.backward()
        meta_grad_vector = get_grad_vector(args, model_temp.parameters, grad_dims)
        add_grad(model.parameters, meta_grad_vector, grad_dims)

        if args.friction:
            get_weight_accumelated_gradient_norm(model)
            weight_gradient_norm(model, args.friction)
        if args.kl_far == -1:
            weight_loss = get_weight_norm_diff(model, task=task, cl=5)
            weight_loss.backward()
        opt.step()

        if 0.5*(loss+b_loss)<0.1 or inner_iter==args.conv_iter :
            #if inner_iter==args.conv_iter :
            #    print("****************inner_iter", inner_iter, float(loss + b_loss))
            return
        inner_iter+=1


def rand_optimize(args, model, opt, input_x, input_y, mem_x, mem_y,task):
    model.zero_grad()
    if args.mask:  # MASK OUT UN Replayed or seen classes
        yy = torch.cat((input_y, mem_y), dim=0)

        mask = get_mask_unused_memories(yy, args.n_classes)
    inner_iter = 1
    while True:

        logits = model(input_x)
        logits_buffer = model(mem_x)
        if args.mask:
            logits = logits.masked_fill(mask, 1e-9)
            logits_buffer = logits_buffer.masked_fill(mask, 1e-9)

        loss = F.cross_entropy(logits, input_y)
        loss.backward()  # Loss of the new batch
        b_loss = (args.multiplier * F.cross_entropy(logits_buffer, mem_y))
        b_loss.backward()  # Loss of the neighbours

        if args.friction:
            get_weight_accumelated_gradient_norm(model)
            weight_gradient_norm(model, args.friction)
        if args.kl_far == -1:
            weight_loss = get_weight_norm_diff(model, task=task, cl=5)
            weight_loss.backward()
        opt.step()

        if 0.5 * (loss + b_loss) < 0.1 or inner_iter == args.conv_iter:
            #if inner_iter == args.conv_iter:
                #print("****************inner_iter", inner_iter, float(loss + b_loss))
            return
        inner_iter += 1
def compute_lossgrad(args,model,input_x,input_y,mem_x,mem_y,far_mem_x):

    if args.mask:  # MASK OUT UN Replayed or seen classes
        yy = torch.cat((input_y, mem_y), dim=0)

        mask = get_mask_unused_memories(yy, args.n_classes)


    logits=model(input_x)
    logits_buffer = model(mem_x)
    if args.mask:
        logits = logits.masked_fill(mask, 1e-9)
        logits_buffer = logits_buffer.masked_fill(mask, 1e-9)

    #those will be already computed if not masking
    #
        loss = F.cross_entropy(logits, input_y)
        loss.backward()  # Loss of the new batch

    b_loss=(args.multiplier * F.cross_entropy(logits_buffer, mem_y))
    b_loss.backward()#Loss of the neighbours
    grad_dims = []
    for param in model.parameters():
        grad_dims.append(param.data.numel())
    model_temp = get_future_step_parameters_with_grads(model,  args.lr)

    KL=- args.kl_far *(F.softmax(model(far_mem_x).detach(), dim=1) * F.log_softmax(model_temp(far_mem_x), dim=1)).sum(dim=1).mean()
    #print("task",task,"loss",float(loss),"b_loss",float(b_loss),"KL value",float(KL))
    KL.backward()
    meta_grad_vector = get_grad_vector(args, model_temp.parameters, grad_dims)
    add_grad(model.parameters, meta_grad_vector, grad_dims)


def get_nearbysamples(args,input_x,input_hidden,b_hidden,b_y):


    if args.balanced:
        return get_blnearbysamples(args,input_x,input_hidden,b_hidden,b_y)
    close_indices = []
    all_dist = torch.zeros(args.subsample).cuda()
    buffer_per_sample = int(args.buffer_batch_size / input_x.size(0))
    added_labels= torch.unique(b_y).cpu().numpy().tolist()
    for sample in input_hidden:
        pdist = torch.nn.PairwiseDistance(p=2)
        input_rep = sample.repeat(b_hidden.size(0), 1)

        dist_to_all = pdist(b_hidden, input_rep)
        try:
            all_dist += dist_to_all
        except:
            pdb.set_trace()
        _, sorted_indices = torch.sort(dist_to_all)
        added = 0

        for new_added_index in sorted_indices:

            if not new_added_index in close_indices and  int(b_y[new_added_index]) in added_labels:

                close_indices.append(new_added_index)
                added_labels.remove(int(b_y[new_added_index]))
                added += 1
                if len(added_labels) ==0:
                    added_labels = torch.unique(b_y).cpu().numpy().tolist()

            if added == buffer_per_sample:
                break

    return close_indices,all_dist


def get_blnearbysamples(args,input_x,input_hidden,b_hidden,b_y):
    close_indices = []
    all_dist = torch.zeros(args.subsample).cuda()
    buffer_per_sample = int(args.buffer_batch_size / input_x.size(0))
    added_labels= torch.unique(b_y).cpu().numpy().tolist()
    for sample in input_hidden:
        pdist = torch.nn.PairwiseDistance(p=2)
        input_rep = sample.repeat(b_hidden.size(0), 1)

        dist_to_all = pdist(b_hidden, input_rep)
        try:
            all_dist += dist_to_all
        except:
            pdb.set_trace()
        _, sorted_indices = torch.sort(dist_to_all)
        added = 0

        for new_added_index in sorted_indices:

            if not new_added_index in close_indices and  int(b_y[new_added_index]) in added_labels:

                close_indices.append(new_added_index)
                added_labels.remove(int(b_y[new_added_index]))
                added += 1
                if len(added_labels) ==0:
                    added_labels = torch.unique(b_y).cpu().numpy().tolist()

            if added == buffer_per_sample:
                break

    return close_indices,all_dist
def get_mask_unused_memories(Y,nb_classes):

    xx = torch.range(0, nb_classes-1).cuda()

    not_in_labels = Y.view(1, -1).eq(xx.view(-1, 1)).sum(1)
    mask = torch.eq(not_in_labels, 0)
    return mask
def get_future_step_parameters_with_grads(this_net,lr=1):
    """
    computes \theta-\delta\theta
    :param this_net:
    :param grad_vector:
    :return:
    """
    new_net=copy.deepcopy(this_net)
    #new_net=this_net
    for name, param in this_net.named_parameters():
        for namex, paramx in new_net.named_parameters():
            if namex==name:

                paramx=param - lr*param.grad
    return new_net

def get_grad_dims(self):
    self.grad_dims = []
    for param in self.net.parameters():
        self.grad_dims.append(param.data.numel())


def weight_gradient_norm(model,friction):
    normalized_friction_weight(model,friction)
    min=1
    if not hasattr(model,"mingr"):
        model.mingr=min
    for name, param in model.named_parameters():

        if not "linear" in name and param.grad is not None:
            if friction==1:
                friction_val=get_friction(param.gr_norm,mu=1)
                if float(torch.min(friction_val))<model.mingr:

                    print("MIN",float(torch.min(friction_val)) )
                    model.mingr=torch.min(friction_val)
                param.grad.data = friction * param.grad.data * friction_val

            else:
                param.grad.data = friction * param.grad.data * param.gr_norm_maxmin
def get_weight_norm_diff(model,beta=1,task=0,cl=5):
    loss = 0
    for name, param in model.named_parameters():
        if "linear" in name:
            for name1, param1 in model.named_parameters():
                if "linear" in name1 and not name1==name:

                    #pdb.set_trace()
                    fullfc=torch.cat((param,param1.unsqueeze(1)),dim=1)
                    norm_fullfc=torch.norm(fullfc,dim=1).unsqueeze(1)
                    norm_fullfc_rep = norm_fullfc.repeat(1, norm_fullfc.size(0))
                    
                    with torch.no_grad():

                        norm_fullfc_rep_t=norm_fullfc.repeat(1,norm_fullfc.size(0)).clone()
                        norm_fullfc_rep_t=norm_fullfc_rep_t.transpose(0,1)
                    loss=torch.dist(norm_fullfc_rep[cl*task:cl*(task+1)],norm_fullfc_rep_t[cl*task:cl*(task+1)])/(norm_fullfc_rep_t[cl*task:cl*(task+1)].size(0)*norm_fullfc_rep_t.size(0))
                    print(loss)
                    return beta * (loss)
            # for neuron_weight in range(param.size(0)):
            #     for other_neuron_weight in range(neuron_weight):
            #         loss+=(1/neuron_weight)*torch.abs(torch.norm(param[neuron_weight])-torch.norm(param[other_neuron_weight]))


def weight_friction(model,friction):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = friction*param.grad.data*get_friction(param.data)

def get_friction(w,mu=1):

    term=torch.exp(mu*w)
    #friction=torch.div(4*term,torch.pow(1+term,2))

    friction = torch.exp(-mu * torch.pow(w,2))


    return friction

def get_weight_accumelated_gradient_norm(model):

    if not hasattr(model,'stepcount'):
        model.stepcount=1
    for w in model.parameters():
        if not hasattr(w,"gr_norm") and  w.grad is not None:
            w.gr_norm=torch.zeros(w.data.size()).cuda()
            print("creating gr_norm")

        if w.grad is not None:
            w.gr_norm =((w.gr_norm*(model.stepcount-1)+ torch.abs(w.grad.data).clone())/model.stepcount)

    model.stepcount += 1

    return model

def normalized_friction_weight(model,friction):
    if friction==2:
        max_accgr=0
        min_accgr= 0
        for w in model.parameters():
            if  hasattr(w,"gr_norm") :
                max_accgr = max(max_accgr, torch.max(w.gr_norm))
                min_accgr = min(min_accgr, torch.min(w.gr_norm))
        for w in model.parameters():
            if  hasattr(w,"gr_norm") :

                w.gr_norm_maxmin=1- ((w.gr_norm-min_accgr)/(max_accgr-min_accgr))

    return model

''' Others '''
def onehot(t, num_classes, device='cpu'):
    """
    convert index tensor into onehot tensor
    :param t: index tensor
    :param num_classes: number of classes
    """
    return torch.zeros(t.size()[0], num_classes).to(device).scatter_(1, t.view(-1, 1), 1)

def distillation_KL_loss(y, teacher_scores, T, scale=1, reduction='batchmean'):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    return F.kl_div(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1),
            reduction=reduction) * scale


def naive_cross_entropy_loss(input, target, size_average=True):
    """
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    input = torch.log(F.softmax(input, dim=1).clamp(1e-5, 1))
    # input = input - torch.log(torch.sum(torch.exp(input), dim=1)).view(-1, 1)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2

def out_mask(t, nc_per_task, n_outputs):
    # make sure we predict classes within the current task
    offset1 = int(t * nc_per_task)
    offset2 = int((t + 1) * nc_per_task)
    if offset1 > 0:
        output[:, :offset1].data.fill_(-10e10)
    if offset2 < self.n_outputs:
        output[:, offset2:n_outputs].data.fill_(-10e10)

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(input.size(0), *self.shape)

''' LOG '''
def logging_per_task(wandb, log, run, mode, metric, task=0, task_t=0, value=0):
    if 'final' in metric:
        log[run][mode][metric] = value
    else:
        log[run][mode][metric][task_t, task] = value

    #print('run {}\t {}\t {}\t task {}\t {:.4f}'.format(run, mode, metric, task, value))

    if wandb is not None:
        if 'final' in metric:
            wandb.log({mode+metric:value}, step=run)

def print_(log, mode, task):
    to_print = mode + ' ' + str(task) + ' '
    for name, value in log.items():
        # only print acc for now
        if len(value) > 0:
            name_ = name + ' ' * (12 - len(name))
            value = sum(value) / len(value)

            if 'acc' in name or 'gen' in name:
                to_print += '{}\t {:.4f}\t'.format(name_, value)
                # print('{}\t {}\t task {}\t {:.4f}'.format(mode, name_, task, value))

    print(to_print)

def get_logger(names, n_runs=1, n_tasks=None):
    log = OD()
    #log = DD()
    log.print_ = lambda a, b: print_(log, a, b)
    for i in range(n_runs):
        log[i] = {}
        for mode in ['train','valid','test']:
            log[i][mode] = {}
            for name in names:
                log[i][mode][name] = np.zeros([n_tasks,n_tasks])

            log[i][mode]['final_acc'] = 0.
            log[i][mode]['final_forget'] = 0.
            log[i][mode]['last_task_acc'] = 0.
            log[i][mode]['allbutfirst_tasks_acc'] = 0.
    return log

def get_temp_logger(exp, names):
    log = OD()
    log.print_ = lambda a, b: print_(log, a, b)
    for name in names: log[name] = []
    return log
