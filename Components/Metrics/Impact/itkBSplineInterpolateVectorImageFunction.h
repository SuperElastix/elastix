/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

/**
 * \class BSplineInterpolateVectorImageFunction
 * \brief Helper class to interpolate each component of a VectorImage using separate B-Spline interpolators.
 *
 * This class enables feature-wise interpolation of ITK VectorImages using one B-Spline interpolator
 * per channel. It supports evaluation at arbitrary physical points and returns the interpolated
 * values or spatial derivatives as Torch tensors for downstream use in workflows.
 */
#ifndef itkBSplineInterpolateVectorImageFunction_h
#define itkBSplineInterpolateVectorImageFunction_h

#include <itkVectorImage.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <vector>
#include <torch/torch.h>

template <typename TImage, typename TInterpolator>
class BSplineInterpolateVectorImageFunction
{
public:
  using ImageType = TImage;
  using InterpolatorType = TInterpolator;
  using PixelType = typename ImageType::PixelType;

  BSplineInterpolateVectorImageFunction() = default;

  /**
   * \brief Initializes one B-Spline interpolator per feature channel in the input VectorImage.
   *
   * Each channel of the vector image is assigned a separate B-Spline interpolator, as ITK's
   * BSplineInterpolateImageFunction does not natively support VectorImages.
   *
   * \param vectorImage The input VectorImage to initialize the B-Spline interpolators for each channel.
   */
  void
  SetInputImage(typename ImageType::Pointer vectorImage);

  /**
   * \brief Interpolates the selected feature channels at a given physical point.
   *
   * \param point The physical coordinate where interpolation is performed.
   * \param subsetOfFeatures Indices of feature channels to interpolate.
   *
   * \return A 1D torch::Tensor containing interpolated values for the requested channels.
   */
  torch::Tensor
  Evaluate(typename ImageType::PointType point, std::vector<unsigned int> subsetOfFeatures) const;

  /**
   * \brief Evaluates the spatial derivative of selected features at a given point.
   *
   * Computes gradients of the selected feature channels with respect to spatial dimensions
   * using the underlying B-Spline interpolators.
   *
   * \param point The physical coordinate at which derivatives are computed.
   * \param subsetOfFeatures Indices of feature channels to differentiate.
   *
   * \return A 2D torch::Tensor (Channels × SpatialDimension) with spatial gradients per feature.
   */
  torch::Tensor
  EvaluateDerivative(typename ImageType::PointType point, std::vector<unsigned int> subsetOfFeatures) const;

private:
  typename ImageType::Pointer                     m_VectorImage;
  std::vector<typename InterpolatorType::Pointer> m_Interpolators;
};

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkBSplineInterpolateVectorImageFunction.hxx"
#endif

#endif // end #ifndef itkBSplineInterpolateVectorImageFunction_h
