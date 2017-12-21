import tensorflow as tf
import numpy as np


DIVISION_ADJUSTMENT = 1e-9

def z_plus_prop(X, W, R, factor):
    V = np.maximum(0, W) + (1 - factor) * np.minimum(0, W)

    Z = np.dot(X, V) + DIVISION_ADJUSTMENT
    S = R / Z
    C = np.dot(S, V.T)
    return X * C


def z_plus_prop_tf(X, W, R, factor=1):
    V = tf.maximum(0.0, W) + (1.0 - factor) * tf.minimum(0.0, W)

    Z = tf.matmul(X, V) + DIVISION_ADJUSTMENT
    S = R / Z
    C = tf.matmul(S, tf.transpose(V))
    return X * C

def z_beta_prop(X, W, R, factor, lowest=-1, highest=1):
    W, V, U = W, np.maximum(0, W), np.minimum(0, W)
    X, L, H = X, X * 0 + lowest, X * 0 + highest

    Z = np.dot(X, W) - factor*(np.dot(L, V) + np.dot(H, U)) + DIVISION_ADJUSTMENT
    S = R / Z
    return X * np.dot(S, W.T) - factor*(L * np.dot(S, V.T) + H * np.dot(S, U.T))

def z_plus_beta_prop(X_p, W_p, X_b, W_b, R, factor, lowest=-1, highest=1):
    V_p = np.maximum(0, W_p) + (1 - factor) * np.minimum(0, W_p)
    Z_p = np.dot(X_p, V_p) + DIVISION_ADJUSTMENT

    W_b, V_b, U_b = W_b, np.maximum(0, W_b), np.minimum(0, W_b)
    L_b, H_b = X_b * 0 + lowest, X_b * 0 + highest
    Z_b = np.dot(X_b, W_b) - factor*(np.dot(L_b, V_b) + np.dot(H_b, U_b)) + DIVISION_ADJUSTMENT

    Z = Z_p + Z_b

    # z-plus
    S_p = R / Z
    C_p = np.dot(S_p, V_p.T);
    R_p = X_p * C_p

    # z-beta
    S_b = R / Z
    R_b = X_b * np.dot(S_b, W_b.T) - factor*(L_b * np.dot(S_b, V_b.T) + H_b * np.dot(S_b, U_b.T))

    return R_p, R_b

# def next_conv_prop(X, W, b):

def pool_prop(x, gradients):
    C = gradients
    return x * C


def pool_prop_tf(x, activations, relevances):
    s = relevances / (activations + DIVISION_ADJUSTMENT)
    c = tf.gradients(activations, x, grad_ys=s)[0]

    return x*c


def pool_prop_test_tf(x, activations, relevances):
    s = relevances / (activations + DIVISION_ADJUSTMENT)

    c = tf.gradients(activations, x, grad_ys=s)[0]
    return c


def conv_prop_tf(X, conv_layer, R):
    pself = conv_layer.clone()
    pself.W = tf.maximum(0.0, pself.W)
    pself.b = 0 * pself.b

    activations, _ = pself.conv(X)
    z = activations + DIVISION_ADJUSTMENT
    s = R / z

    shape_x = tf.shape(X)

    c = tf.nn.conv2d_backprop_input(shape_x, pself.W,
                                    out_backprop=s,
                                    strides=pself.strides,
                                    padding=pself.padding
                                    )

    return X * c


def conv_beta_prop_tf(X, conv_layer, R, lowest=-1.0, highest=1.0):
    iself = conv_layer.clone()
    iself.b = 0.0 * iself.b

    nself = conv_layer.clone()
    nself.b = 0.0 * nself.b
    nself.W = tf.minimum(0.0, nself.W)

    pself = conv_layer.clone()
    pself.b = 0.0 * pself.b
    pself.W = tf.maximum(0.0, pself.W)

    X,L,H = X, X*0.0+lowest, X*0+highest

    i_act, _ = iself.conv(X)
    p_act, _ = pself.conv(L)
    n_act, _ = nself.conv(H)

    Z = i_act - p_act - n_act + DIVISION_ADJUSTMENT

    S = R/Z

    shape_x = tf.shape(X)

    grad_params = dict(
        out_backprop=S,
        strides=conv_layer.strides,
        padding=conv_layer.padding
    )

    R = X*tf.nn.conv2d_backprop_input(shape_x, iself.W, **grad_params) - \
        L*tf.nn.conv2d_backprop_input(shape_x, pself.W, **grad_params) - \
        H*tf.nn.conv2d_backprop_input(shape_x, nself.W, **grad_params) \

    return R


