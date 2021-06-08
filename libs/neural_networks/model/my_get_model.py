
def get_model(model_name, num_class=2):
    if model_name == 'Cls_3d':
        from libs.neural_networks.model.cls_3d import Cls_3d
        model = Cls_3d(n_class=num_class)

    # region medical net
    if model_name == 'medical_net_resnet34':
        from libs.neural_networks.model.MedicalNet.resnet import resnet34, Resnet3d_cls
        base_model = resnet34(output_type='classification')
        model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='BasicBlock', add_dense1=True)
    if model_name == 'medical_net_resnet50':
        from libs.neural_networks.model.MedicalNet.resnet import resnet50, Resnet3d_cls
        base_model = resnet50(output_type='classification')
        model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='Bottleneck', add_dense1=True)
    if model_name == 'medical_net_resnet101':
        from libs.neural_networks.model.MedicalNet.resnet import resnet101, Resnet3d_cls
        base_model = resnet101(output_type='classification')
        model = Resnet3d_cls(base_model=base_model, n_class=num_class, block_type='Bottleneck', add_dense1=True)
    # endregion

    '''
    if model_name == 'ModelsGenesis':
        from libs.neural_networks.model.ModelsGenesis.unet3d import UNet3D, TargetNet
        base_model = UNet3D()
        model = TargetNet(base_model, n_class=num_class)
    '''

    '''
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
    '''

    return model