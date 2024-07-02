from utils import *

if __name__ == '__main__':
    # 随机种子和cuda配置
    # torch.manual_seed(0)
    device = get_device()
    print('running in {}'.format(device))
    torch.backends.cudnn.benchmark = True  # 使用cudnn加速卷积运算

    # 加载数据集
    train_dataloader, test_dataloader = get_cifar10_data(batch_size=64)

    # 创建教师模型
    t_model = torchvision.models.resnet34()  # 实例化
    t_model = t_model.to(device)  # 指定到device

    # 定义损失函数和优化器
    epoch = 3
    lr = 1e-4
    criterion = nn.CrossEntropyLoss()
    t_optimizer = torch.optim.Adam(t_model.parameters(), lr=lr)

    # 训练
    print("""------------------------------正常训练教师模型---------------------------------""")
    t_time = train(t_model, train_dataloader, criterion, t_optimizer, device, epochs=epoch)
    t_acc, _ = time_model_evaluation(t_model, test_dataloader, device)

    # 这部分仅仅是为了展示单独训练一个学生模型时的效果，与采用蒸馏训练对比一下
    print("""------------------------------正常训练学生模型----------------------------------""")
    nor_s_model = torchvision.models.resnet18()
    nor_s_model = nor_s_model.to(device)

    nor_s_optimizer = torch.optim.Adam(nor_s_model.parameters(), lr=lr)
    nor_s_time = train(nor_s_model, train_dataloader, criterion, nor_s_optimizer, device, epochs=epoch)
    nor_s_acc, _ = time_model_evaluation(nor_s_model, test_dataloader, device)

    print("""------------------------------蒸 馏----------------------------------""")
    s_kd_model = torchvision.models.resnet18()  # 准备新的学生模型
    s_kd_model = s_kd_model.to(device)

    # soft_loss:0.3, hard_loss:0.7, temp:7
    kd_criterion = make_criterion(alpha=0.3, T=7, mode='kl')
    s_kd_optimizer = torch.optim.Adam(s_kd_model.parameters(), lr=1e-4)

    s_kd_time = distillation_train(s_kd_model, t_model, train_dataloader, device, kd_criterion, s_kd_optimizer, epochs=epoch)
    s_kd_acc, _ = time_model_evaluation(s_kd_model, test_dataloader, device)

    print('教师模型大小：{}M'.format(print_size_of_model(t_model)))
    print('训练消耗时间：{0:.3f}s, 预测精度：{1:.3f}'.format(t_time, t_acc))

    print('正常训练学生模型大小：{}M'.format(print_size_of_model(nor_s_model)))
    print('训练消耗时间：{0:.3f}s, 预测精度：{1:.3f}'.format(nor_s_time, nor_s_acc))

    print('蒸馏学生模型大小：{}M'.format(print_size_of_model(s_kd_model)))
    print('训练消耗时间：{0:.3f}s, 预测精度：{1:.3f}'.format(s_kd_time, s_kd_acc))

    """ alpha=0.3, T=7, mode='kl'
            教师模型大小：87.315428M
            训练消耗时间：1678.092s, 预测精度：73.660
            正常训练学生模型大小：46.828292M
            训练消耗时间：1084.239s, 预测精度：68.290
            蒸馏学生模型大小：46.828292M
            训练消耗时间：3669.204s, 预测精度：72.780
            """