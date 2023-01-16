import jax.numpy as jnp
import jax
import optax


def reconstruction(feature_matrix, kernels):
    convs = jnp.zeros((len(feature_matrix),))

    def body(carry, kernel):
        convs, i = carry
        convs += jnp.convolve(feature_matrix[:, i], kernel, mode="same")
        # convs += jax.scipy.signal.convolve(feature_matrix[:,i], kernel, mode="same")
        return (convs, i + 1), None

    (convs, _), _ = jax.lax.scan(body, (convs, 0), kernels)
    return convs / float(len(kernels))


# more flexible reconstruction that allows for dilation and stride
def reconstruction_gen(feature_matrix, kernels, dilations):
    kernel_len = kernels.shape[1]
    nsamples = len(feature_matrix)
    convs = jnp.zeros((nsamples,))
    dn = jax.lax.conv_dimension_numbers([1, nsamples, 1], [kernel_len, 1, 1], ("NHC", "HIO", "NHC"))
    for i, (kernel, dilation) in enumerate(zip(kernels, dilations)):
        _result = jax.lax.conv_general_dilated(
            feature_matrix[:, i][None, :, None],  # lhs = image tensor
            kernel[::-1][:, None, None],  # rhs = conv kernel tensor
            (1,),  # window strides
            "SAME",  # padding mode
            (1,),  # lhs/image dilation
            (dilation,),  # rhs/kernel dilation
            dn,
        )  # dimension_numbers = lhs, rhs, out dimension permutation
        # _result = _recon_conv(feature_matrix[:, i][None, :, None], kernel[::-1][:, None, None], int(dilation), dn)
        convs += _result.squeeze()
    return convs / float(len(kernels))


def kernel_loss(dlight_trace, feature_matrix, kernels):
    # try to reconstruct dlight with kernels (include an offset maybe?)
    recon = reconstruction(feature_matrix, kernels)
    # return jnp.mean((dlight_trace - recon) ** 2)
    return jnp.mean(optax.huber_loss(recon, dlight_trace))


def kernel_loss_smooth(dlight_trace, feature_matrix, kernels):
    # try to reconstruct dlight with kernels (include an offset maybe?)
    recon = reconstruction(feature_matrix, kernels)
    # sharpness = jnp.max(jnp.diff(kernels,axis=1,n=2) ** 2,axis=1)
    sharpness = (jnp.diff(kernels, axis=0) ** 2).sum(axis=0)
    return jnp.mean((dlight_trace - recon) ** 2) + sharpness.mean()


def kernel_loss_spline(dlight_trace, feature_matrix, coeffs, basis):
    kernels = basis.dot(coeffs).T
    recon = reconstruction(feature_matrix, kernels)
    return jnp.mean(optax.huber_loss(recon, dlight_trace))
    # return jnp.mean((dlight_trace - recon) ** 2)


def kernel_loss_spline_gen(dlight_trace, feature_matrix, coeffs, basis, dilations):
    kernels = basis.dot(coeffs).T
    # recon = reconstruction(dlight_trace, feature_matrix, kernels, dilations)
    recon = reconstruction_gen(feature_matrix, kernels, dilations)
    return jnp.mean(optax.huber_loss(recon, dlight_trace))
    # return jnp.mean((dlight_trace - recon) ** 2)
