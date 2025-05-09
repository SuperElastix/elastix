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

#ifndef _itkBSplineInterpolateVectorImageFunction_hxx
#define _itkBSplineInterpolateVectorImageFunction_hxx

#include "itkBSplineInterpolateVectorImageFunction.h"
#include <itkVectorIndexSelectionCastImageFilter.h>

/**
 * ******************* SetInputImage ***********************
 */
template <typename TImage, typename TInterpolator>
void
BSplineInterpolateVectorImageFunction<TImage, TInterpolator>::SetInputImage(typename TImage::Pointer vectorImage)
{
  // Loop over each feature (channel) in the vector image
  // Create a separate scalar image and corresponding interpolator for it
  for (unsigned int i = 0; i < vectorImage->GetVectorLength(); ++i)
  {
    auto selector = itk::VectorIndexSelectionCastImageFilter<TImage, itk::Image<float, TImage::ImageDimension>>::New();
    selector->SetInput(vectorImage);
    selector->SetIndex(i);
    selector->Update();

    auto interpolator = TInterpolator::New();
    interpolator->SetInputImage(selector->GetOutput());
    interpolator->SetSplineOrder(3);
    this->m_Interpolators.push_back(interpolator);
  }
} // end SetInputImage

/**
 * ******************* Evaluate ***********************
 */
template <typename TImage, typename TInterpolator>
typename torch::Tensor
BSplineInterpolateVectorImageFunction<TImage, TInterpolator>::Evaluate(typename TImage::PointType point,
                                                                       std::vector<unsigned int> subsetOfFeatures) const
{
  std::vector<float> result;
  for (const unsigned int feature : subsetOfFeatures)
  {
    result.push_back(this->m_Interpolators[feature]->Evaluate(point));
  }
  return torch::from_blob(result.data(), { static_cast<int64_t>(result.size()) }, torch::kFloat32).clone();
} // end Evaluate

/**
 * ******************* EvaluateDerivative ***********************
 */
template <typename TImage, typename TInterpolator>
typename torch::Tensor
BSplineInterpolateVectorImageFunction<TImage, TInterpolator>::EvaluateDerivative(
  typename ImageType::PointType point,
  std::vector<unsigned int>     subsetOfFeatures) const
{
  using CovariantVectorType = itk::CovariantVector<float, TImage::ImageDimension>;

  std::vector<float>  derivative(subsetOfFeatures.size() * TImage::ImageDimension, 0.0f);
  CovariantVectorType dev;
  // Fill the derivative tensor with directional gradients for each selected feature
  for (int i = 0; i < subsetOfFeatures.size(); ++i)
  {
    dev = this->m_Interpolators[subsetOfFeatures[i]]->EvaluateDerivative(point);
    for (unsigned int it = 0; it < TImage::ImageDimension; ++it)
    {
      derivative[i * TImage::ImageDimension + it] = static_cast<float>(dev[it]);
    }
  }
  return torch::from_blob(derivative.data(),
                          { static_cast<int64_t>(subsetOfFeatures.size()), TImage::ImageDimension },
                          torch::kFloat32)
    .clone();
} // end EvaluateDerivative

#endif // end #ifndef _itkBSplineInterpolateVectorImageFunction_hxx
