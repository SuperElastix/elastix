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
#ifndef _itkUpsampleBSplineParametersFilter_hxx
#define _itkUpsampleBSplineParametersFilter_hxx

#include "itkUpsampleBSplineParametersFilter.h"

#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineDecompositionImageFilter.h"
#include "itkResampleImageFilter.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TArray, class TImage>
UpsampleBSplineParametersFilter<TArray, TImage>::UpsampleBSplineParametersFilter()
{
  this->m_BSplineOrder = 3;

  // Initialize grid settings.
  this->m_CurrentGridOrigin.Fill(0.0);
  this->m_CurrentGridSpacing.Fill(0.0);
  this->m_CurrentGridDirection.Fill(0.0);
  this->m_RequiredGridOrigin.Fill(0.0);
  this->m_RequiredGridSpacing.Fill(0.0);
  this->m_RequiredGridDirection.Fill(0.0);
} // end Constructor()


/**
 * ******************* UpsampleParameters *******************
 */

template <class TArray, class TImage>
void
UpsampleBSplineParametersFilter<TArray, TImage>::UpsampleParameters(const ArrayType & parameters_in,
                                                                    ArrayType &       parameters_out)
{
  /** Determine if upsampling is required. */
  if (!this->DoUpsampling())
  {
    parameters_out = parameters_in; // \todo: hard copy, can be avoided
    return;
  }

  /** Typedefs. */
  using UpsampleFilterType = itk::ResampleImageFilter<ImageType, ImageType>;
  using CoefficientUpsampleFunctionType = itk::BSplineResampleImageFunction<ImageType, ValueType>;
  using DecompositionFilterType = itk::BSplineDecompositionImageFilter<ImageType, ImageType>;

  /** Get the number of parameters. */
  const unsigned int currentNumberOfPixels = this->m_CurrentGridRegion.GetNumberOfPixels();
  const unsigned int requiredNumberOfPixels = this->m_RequiredGridRegion.GetNumberOfPixels();

  /** Create the new vector of output parameters, with the correct size. */
  parameters_out.SetSize(requiredNumberOfPixels * Dimension);

  /** Get the pointer to the data of the input parameters. */
  PixelType * inputDataPointer = const_cast<PixelType *>(parameters_in.data_block());
  PixelType * outputDataPointer = const_cast<PixelType *>(parameters_out.data_block());

  /** The input parameters are represented as a coefficient image. */
  ImagePointer coeffs_in = ImageType::New();
  coeffs_in->SetOrigin(this->m_CurrentGridOrigin);
  coeffs_in->SetSpacing(this->m_CurrentGridSpacing);
  coeffs_in->SetDirection(this->m_CurrentGridDirection);
  coeffs_in->SetRegions(this->m_CurrentGridRegion);

  /** Loop over dimension: each direction is upsampled separately. */
  for (unsigned int j = 0; j < Dimension; ++j)
  {
    /** Fill the coefficient image with parameter data. */
    coeffs_in->GetPixelContainer()->SetImportPointer(inputDataPointer, currentNumberOfPixels);
    inputDataPointer += currentNumberOfPixels;

    /** Set the coefficient image as the input of the upsampler filter.
     * The upsampler samples the deformation field at the locations
     * of the new control points, given the current coefficients
     * (note: it does not just interpolate the coefficient image,
     * which would be wrong). The B-spline coefficients that
     * describe the resulting image are computed by the
     * decomposition filter.
     *
     * This code is derived from the itk-example DeformableRegistration6.cxx.
     */
    auto upsampler = UpsampleFilterType::New();
    auto coeffUpsampleFunction = CoefficientUpsampleFunctionType::New();
    auto decompositionFilter = DecompositionFilterType::New();

    /** Setup the upsampler. */
    upsampler->SetInterpolator(coeffUpsampleFunction);
    upsampler->SetSize(this->m_RequiredGridRegion.GetSize());
    upsampler->SetOutputStartIndex(this->m_RequiredGridRegion.GetIndex());
    upsampler->SetOutputSpacing(this->m_RequiredGridSpacing);
    upsampler->SetOutputOrigin(this->m_RequiredGridOrigin);
    upsampler->SetOutputDirection(this->m_RequiredGridDirection);
    upsampler->SetInput(coeffs_in);

    /** Setup the decomposition filter. */
    decompositionFilter->SetSplineOrder(this->m_BSplineOrder);
    decompositionFilter->SetInput(upsampler->GetOutput());

    /** Do the upsampling. */
    try
    {
      decompositionFilter->UpdateLargestPossibleRegion();
      // \todo: the decomposition filter could be multi-threaded
      // by deriving it from the RecursiveSeparableImageFilter,
      // similar to the SmoothingRecursiveGaussianImageFilter.
    }
    catch (itk::ExceptionObject & excp)
    {
      /** Add information to the exception. */
      excp.SetLocation("UpsampleBSplineParametersFilter - UpsampleParameters()");
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while using decompositionFilter.\n";
      excp.SetDescription(err_str);

      /** Pass the exception to an higher level. */
      throw;
    }

    /** Get a pointer to the upsampled coefficient image. */
    const PixelType * coeffs_out = decompositionFilter->GetOutput()->GetBufferPointer();

    /** Copy the contents of coeffs_out in a ParametersType array. */
    std::copy(coeffs_out, coeffs_out + requiredNumberOfPixels, outputDataPointer + requiredNumberOfPixels * j);

  } // end for dimension loop

} // end UpsampleParameters()


/**
 * ******************* DoUpsampling *******************
 */

template <class TArray, class TImage>
bool
UpsampleBSplineParametersFilter<TArray, TImage>::DoUpsampling()
{
  bool ret = (this->m_CurrentGridOrigin != this->m_RequiredGridOrigin);
  ret |= (this->m_CurrentGridSpacing != this->m_RequiredGridSpacing);
  ret |= (this->m_CurrentGridDirection != this->m_RequiredGridDirection);
  ret |= (this->m_CurrentGridRegion != this->m_RequiredGridRegion);

  return ret;

} // end DoUpsampling()


/**
 * ******************* PrintSelf *******************
 */

template <class TArray, class TImage>
void
UpsampleBSplineParametersFilter<TArray, TImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "CurrentGridOrigin: " << this->m_CurrentGridOrigin << std::endl;
  os << indent << "CurrentGridSpacing: " << this->m_CurrentGridSpacing << std::endl;
  os << indent << "CurrentGridDirection: " << this->m_CurrentGridDirection << std::endl;
  os << indent << "CurrentGridRegion: " << this->m_CurrentGridRegion << std::endl;

  os << indent << "RequiredGridOrigin: " << this->m_RequiredGridOrigin << std::endl;
  os << indent << "RequiredGridSpacing: " << this->m_RequiredGridSpacing << std::endl;
  os << indent << "RequiredGridDirection: " << this->m_RequiredGridDirection << std::endl;
  os << indent << "RequiredGridRegion: " << this->m_RequiredGridRegion << std::endl;

  os << indent << "BSplineOrder: " << this->m_BSplineOrder << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif
