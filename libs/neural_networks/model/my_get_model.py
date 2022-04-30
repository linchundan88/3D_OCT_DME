import torch

def get_model(model_name, num_class=2, model_file=None, **params):
    if 'drop_prob' not in params:
        drop_prob = 0
    else:
        drop_prob = params['drop_prob']

    if model_name == 'cls_3d':
        from libs.neural_networks.model.cls_3d.cls_3d import Cls_3d
        model = Cls_3d(n_class=num_class, dropout_prob=drop_prob)

    if model_name == 'ModelsGenesis':
        from libs.neural_networks.model.ModelsGenesis.unet3d import UNet3D, TargetNet
        base_model = UNet3D()
        model = TargetNet(base_model, n_class=num_class)

    # region medical net
    if model_name == 'medical_net_resnet34':
        from libs.neural_networks.model.MedicalNet.resnet import resnet34, Resnet3d_cls
        base_model = resnet34(output_type='classification')
        model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='BasicBlock',
                             add_dense1=True, dropout_prob=drop_prob)
    if model_name == 'medical_net_resnet50':
        from libs.neural_networks.model.MedicalNet.resnet import resnet50, Resnet3d_cls
        base_model = resnet50(output_type='classification')
        model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='Bottleneck',
                             add_dense1=True, dropout_prob=drop_prob)
    if model_name == 'medical_net_resnet101':
        from libs.neural_networks.model.MedicalNet.resnet import resnet101, Resnet3d_cls
        base_model = resnet101(output_type='classification')
        model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='Bottleneck',
                             add_dense1=True, dropout_prob=drop_prob)
    # endregion


    # region 3D ResNet  [10, 18, 34, 50, 101, 152, 200]
    from libs.neural_networks.model.model_3d.resnet import generate_model
    
    if model_name == 'resnet18':
        model = generate_model(model_depth=18, n_classes=num_class, n_input_channels=1)
    if model_name == 'resnet34':
        model = generate_model(model_depth=32, n_classes=num_class, n_input_channels=1)
    if model_name == 'resnet50':
        model = generate_model(model_depth=50, n_classes=num_class, n_input_channels=1)
    if model_name == 'resnet101':
        model = generate_model(model_depth=101, n_classes=num_class, n_input_channels=1)
    # endregion


    if model_file is not None:
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    return model