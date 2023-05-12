"""
Author: boren li <borenli@cea-igp.ac.cn>
"""
from keras.models import Model
from keras.layers import Input, concatenate, Dropout, Cropping2D, ZeroPadding2D, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Activation
from keras.regularizers import l2

import ipdb as pdb

"""
U-net, Nest-Net model from: 
https://github.com/MrGiovanni/Nested-UNet/blob/master/model_logic.py
"""

########################################
# 2D Standard
########################################
def standard_unit(input_tensor, stage, nb_filter, kernel_size=(1, 7)):   # modified by perrin
    dropout_rate = 0.2
    act = "relu"

    x = Conv2D(nb_filter, kernel_size, activation=act, name='conv' + stage + '_1',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = BatchNormalization()(x)  # add 
#     x = BatchNormalization(trainable=False)(x, training=True)  # add 2021.8.26        
    x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
    x = Conv2D(nb_filter, kernel_size, activation=act, name='conv' + stage + '_2',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)   # add 
    x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)

    return x



# 版本2： 有大小校验
def crop_and_concat(net_small, net_big, name='merge', axis=-1):
    """
    the size(net_small) <= size(net_big)
    
    :param net_small: The small size of the net/layer.
    :param net_big: The bigger size fo the net/Layer.
    """
    _, net1_height, net1_width, _ = list(net_small._keras_shape)
    _, net2_height, net2_width, _ = list(net_big._keras_shape)    

    h_crop = net2_height - net1_height
    w_crop = net2_width - net1_width
#     pdb.set_trace()
    assert h_crop >= 0
    assert w_crop >= 0 
    if h_crop == 0 and w_crop == 0:
        net2_resize = net_big
    else:
        cropping = ((h_crop//2, h_crop - h_crop//2), (w_crop//2, w_crop - w_crop//2))
        net2_resize = Cropping2D(cropping=cropping)(net_big)
        
#     return net2_resize
    merged_net = concatenate([net_small, net2_resize], name=name, axis=axis)
    return merged_net



def crops_and_concats(net_smallest, nets2concat, name='merge', axis=-1):
    """
    the size(net_small) <= size(every of nets2concat)
    
    :param net_smallest: The small size of the net/layer.
    :param nets2concat: The list of the net/Layer which to be croped.
    """
    if not isinstance(nets2concat, list):
        return net_smallest
    
    # Append the smallest net to the concated list
    list_concated_net = []
    list_concated_net.append(net_smallest)
    
    # Crop the bigger nets
    net0_shape = list(net_smallest._keras_shape)
    for net in nets2concat:
        net_shape = list(net._keras_shape)
    
        top_crop = (net_shape[1] - net0_shape[1]) // 2
        left_crop = (net_shape[2] - net0_shape[2]) // 2
        bottom_crop = net_shape[1] - net0_shape[1] - top_crop
        right_crop =  net_shape[2] - net0_shape[2] - left_crop
        net_resize = Cropping2D(cropping=((top_crop, bottom_crop), (left_crop,right_crop)))(net)
        list_concated_net.append(net_resize)
    
    merged_net = concatenate(list_concated_net, name=name, axis=axis)
    return merged_net

########################################

"""
Standard U-Net [Ronneberger et.al, 2015]
Total params: 7,759,521
"""
def U_Net(img_rows, img_cols, color_type=1, num_class=1):
    nb_filter = [8, 16, 32, 64, 128]
    # nb_filter = [32, 64, 128, 256, 512]
    pool_size = (1, 2)
    padding_size = ((0, 0), (3, 4))
    kernel_size=(1, 7)

    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    zpad = ZeroPadding2D(padding_size)(img_input)

#     pdb.set_trace()
    conv1_1 = standard_unit(zpad, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D(pool_size=pool_size, name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D(pool_size=pool_size, name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D(pool_size=pool_size, name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D(pool_size=pool_size, name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], 3, strides=pool_size, name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], 3, strides=pool_size, name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=3)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], 3, strides=pool_size, name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=3)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], 3, strides=pool_size, name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=3)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal',
                         padding='same', kernel_regularizer=l2(3e-4))(conv1_5)
    crop = Cropping2D(padding_size)(unet_output)

#     pdb.set_trace()
    model = Model(inputs=img_input, outputs=crop)

    return model



"""
Standard UNet++ [Zhou et.al, 2018]
Total params: 9,041,601
"""
def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):
    # nb_filter = [8, 16, 32, 64, 128]
    nb_filter = [32, 64, 128, 256, 512]
    pool_size = (1, 2)
    padding_size = ((0, 0), (3, 4))      

    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    zpad = ZeroPadding2D(padding_size)(img_input)   # padding=((0, 0), (3, 4))表左侧补3个0，右侧补4个0

    conv1_1 = standard_unit(zpad, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D(pool_size=pool_size, name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D(pool_size=pool_size, name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], pool_size, strides=pool_size, name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=3)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D(pool_size=pool_size, name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], pool_size, strides=pool_size, name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=3)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], pool_size, strides=pool_size, name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=3)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D(pool_size=pool_size, name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], pool_size, strides=pool_size, name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=3)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], pool_size, strides=pool_size, name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=3)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], pool_size, strides=pool_size, name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=3)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], pool_size, strides=pool_size, name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], pool_size, strides=pool_size, name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], pool_size, strides=pool_size, name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=3)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], pool_size, strides=pool_size, name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=3)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    crop1 = Cropping2D(padding_size)(nestnet_output_1)
    crop2 = Cropping2D(padding_size)(nestnet_output_2)
    crop3 = Cropping2D(padding_size)(nestnet_output_3)
    crop4 = Cropping2D(padding_size)(nestnet_output_4)

    if deep_supervision:
        model = Model(inputs=img_input, outputs=[crop1, crop2, crop3, crop4])
    else:
        model = Model(inputs=img_input, outputs=[crop4])

    return model



def UNetPlusPlusPro(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):
    """
    Modified UNetPlusPlus. including:  
    crop_and_concat - add function.
    Conv+stride - replace MaxPooling2D with strided convolution.
    """
    nb_filter = [8, 16, 32, 64, 128]  # default
    pool_size = (1, 4)    # default
    kernel_size = (1, 7)

    bn_axis = 3
    img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = Conv2D(nb_filter=nb_filter[0], kernel_size=kernel_size, strides=pool_size, activation='relu', name='pool_conv1', padding="same")(conv1_1)
    
    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = Conv2D(nb_filter=nb_filter[1], kernel_size=kernel_size, strides=pool_size, activation='relu', name='pool_conv2', padding="same")(conv2_1)
    
    up1_2 = Conv2DTranspose(nb_filter[0], kernel_size, strides=pool_size, name='up12', padding='same')(conv2_1)
    conv1_2 = crops_and_concats(conv1_1, [up1_2], name='merge12', axis=bn_axis)    
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = Conv2D(nb_filter=nb_filter[2], kernel_size=kernel_size, strides=pool_size, activation='relu', name='pool_conv3', padding="same")(conv3_1)
    
    up2_2 = Conv2DTranspose(nb_filter[1], kernel_size, strides=pool_size, name='up22', padding='same')(conv3_1)
    conv2_2 = crops_and_concats(conv2_1, [up2_2], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], kernel_size, strides=pool_size, name='up13', padding='same')(conv2_2)
    conv1_3 = crops_and_concats(conv1_1, [up1_3, conv1_2], name='merge13', axis=bn_axis)    
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = Conv2D(nb_filter=nb_filter[3], kernel_size=kernel_size, strides=pool_size, activation='relu', name='pool_conv4', padding="same")(conv4_1) 
    
    up3_2 = Conv2DTranspose(nb_filter[2], kernel_size, strides=pool_size, name='up32', padding='same')(conv4_1)
    conv3_2 = crops_and_concats(conv3_1, [up3_2], name='merge32', axis=bn_axis)        
    
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], kernel_size, strides=pool_size, name='up23', padding='same')(conv3_2)
    conv2_3= crops_and_concats(conv2_1, [up2_3, conv2_2], name='merge23', axis=bn_axis) 
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], kernel_size, strides=pool_size, name='up14', padding='same')(conv2_3)
    conv1_4 = crops_and_concats(conv1_1, [up1_4, conv1_2, conv1_3], name='merge14', axis=bn_axis) 
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])    
    
    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4]) 

    up4_2 = Conv2DTranspose(nb_filter[3], kernel_size, strides=pool_size, name='up42', padding='same')(conv5_1)
    conv4_2 = crops_and_concats(conv4_1, [up4_2], name='merge42', axis=bn_axis)     
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], kernel_size, strides=pool_size, name='up33', padding='same')(conv4_2)
    conv3_3 = crops_and_concats(conv3_1, [up3_3, conv3_2], name='merge33', axis=bn_axis)     
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], kernel_size, strides=pool_size, name='up24', padding='same')(conv3_3)
    conv2_4 = crops_and_concats(conv2_1, [up2_4, conv2_2, conv2_3], name='merge24', axis=bn_axis)     
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], kernel_size, strides=pool_size, name='up15', padding='same')(conv2_4)
    conv1_5 = crops_and_concats(conv1_1, [up1_5, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)       
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)
    
    if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(input=img_input, output=[nestnet_output_4])
    
    return model  


          
def _lr_schedule(epoch):
    """ Learning rate is scheduled to be reduced after 40, 60, 80, 90 epochs.
    """  
    lr = 1e-3
    if epoch > 80:
        lr *= 0.5e-3     
    elif epoch > 45:    # 45, 50, 60
        lr *= 1e-3
    elif epoch > 30:    # 30, 40
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1        
    print('Learning rate: ', lr)
    return lr


if __name__ == '__main__':
#     model = Nest_Net(1, 3001, 1)
#     model.summary()

#     model = unet(1, 3001, 1)
#     model.summary()
