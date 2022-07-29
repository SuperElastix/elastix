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

#ifndef itkVectorMeanDiffusionImageFilter_hxx
#define itkVectorMeanDiffusionImageFilter_hxx

#include "itkVectorMeanDiffusionImageFilter.h"

#include "itkNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "itkProgressReporter.h"

namespace itk
{

/**
 * *********************** Constructor **************************
 */

template <class TInputImage, class TGrayValueImage>
VectorMeanDiffusionImageFilter<TInputImage, TGrayValueImage>::VectorMeanDiffusionImageFilter()
{
  /** Initialize things for the filter. */
  this->m_NumberOfIterations = 0;
  this->m_Radius.Fill(1);
  this->m_RescaleFilter = nullptr;
  this->m_GrayValueImage = nullptr;
  this->m_Cx = nullptr;

} // end Constructor


/**
 * *************** GenerateInputRequestedRegion *****************
 */

template <class TInputImage, class TGrayValueImage>
void
VectorMeanDiffusionImageFilter<TInputImage, TGrayValueImage>::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // get pointers to the input and output
  typename Superclass::InputImagePointer  inputPtr = const_cast<TInputImage *>(this->GetInput());
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();

  if (!inputPtr || !outputPtr)
  {
    return;
  }

  // get a copy of the input requested region (should equal the output
  // requested region)
  typename TInputImage::RegionType inputRequestedRegion;
  inputRequestedRegion = inputPtr->GetRequestedRegion();

  // pad the input requested region by the operator radius
  inputRequestedRegion.PadByRadius(this->m_Radius);

  // crop the input requested region at the input's largest possible region
  if (inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion()))
  {
    inputPtr->SetRequestedRegion(inputRequestedRegion);
    return;
  }
  else
  {
    // Couldn't crop the region (requested region is outside the largest
    // possible region).  Throw an exception.

    // store what we tried to request (prior to trying to crop)
    inputPtr->SetRequestedRegion(inputRequestedRegion);

    // build an exception
    InvalidRequestedRegionError e(__FILE__, __LINE__);
    std::ostringstream          msg;
    msg << static_cast<const char *>(this->GetNameOfClass()) << "::GenerateInputRequestedRegion()";
    e.SetLocation(msg.str().c_str());
    e.SetDescription("Requested region is (at least partially) outside the largest possible region.");
    e.SetDataObject(inputPtr);
    throw e;
  }

} // end GenerateInputRequestedRegion()


/**
 * ********************** GenerateData **************************
 */

template <class TInputImage, class TGrayValueImage>
void
VectorMeanDiffusionImageFilter<TInputImage, TGrayValueImage>::GenerateData()
{
  // \todo to avoid all the copying we can create a vector of
  // outputs of size this->GetNumberOfIterations(). The last
  // output should point to this->GetOutput(). In order to avoid
  // a big memory footprint, at some iteration k, memory of the
  // former iteration should be released: output[ k - 1 ] = 0,
  // and memory of the current iteration should only now be created:
  // output[ k ] = InputImageType::New(); ...; output[ k ]->Allocate().
  // Only for the last iteration: output[ k ]( this->GetOutput() ).

  /** Create feature image. */
  this->FilterGrayValueImage();

  /** Declare things. */
  unsigned int                                      i, j;
  ZeroFluxNeumannBoundaryCondition<InputImageType>  nbc;
  ZeroFluxNeumannBoundaryCondition<DoubleImageType> nbc2;
  NeighborhoodIterator<InputImageType>              nit;
  NeighborhoodIterator<DoubleImageType>             nit2;
  VectorRealType                                    sum;

  /** Allocate output. */
  typename InputImageType::ConstPointer input(this->GetInput());
  typename InputImageType::Pointer      output(this->GetOutput());
  auto                                  outputtmp = InputImageType::New();
  output->SetRegions(input->GetLargestPossibleRegion());

  try
  {
    output->Allocate();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception and throw again. */
    excp.SetLocation("VectorMeanDiffusionImageFilter - GenerateData()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while allocating the filter output.\n";
    excp.SetDescription(err_str);
    throw;
  }

  /** Allocate a temporary output image. */
  outputtmp->SetSpacing(input->GetSpacing());
  outputtmp->SetOrigin(input->GetOrigin());
  outputtmp->SetRegions(input->GetLargestPossibleRegion());

  try
  {
    outputtmp->Allocate();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception and throw again. */
    excp.SetLocation("VectorMeanDiffusionImageFilter - GenerateData()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while allocating a temporary copy.\n";
    excp.SetDescription(err_str);
    throw;
  }

  // support progress methods/callbacks
  // ProgressReporter progress( this, threadId, outputRegionForThread.GetNumberOfPixels() );

  /** Copy input to output. */
  ImageRegionConstIterator<InputImageType> in_it(input, input->GetLargestPossibleRegion());
  ImageRegionIterator<InputImageType>      out_it(output, input->GetLargestPossibleRegion());
  in_it.GoToBegin();
  out_it.GoToBegin();
  while (!in_it.IsAtEnd())
  {
    out_it.Set(in_it.Get());
    ++in_it;
    ++out_it;
  }

  /** Setup neighborhood iterator for the output deformation image. */
  nit = NeighborhoodIterator<InputImageType>(this->m_Radius, output, output->GetLargestPossibleRegion());
  unsigned int neighborhoodSize = nit.Size();
  nit.OverrideBoundaryCondition(&nbc);

  /** Setup neighborhood iterator for the "stiffness coefficient" image. */
  nit2 = NeighborhoodIterator<DoubleImageType>(this->m_Radius, this->m_Cx, this->m_Cx->GetLargestPossibleRegion());
  nit2.OverrideBoundaryCondition(&nbc2);

  /** Setup iterator over outputtmp. */
  ImageRegionIterator<InputImageType> oit(outputtmp, input->GetLargestPossibleRegion());

  /** Initialize c and ci. */
  double c = 0.0;
  double ci = 0.0;

  /** Loop over the number of iterations. */
  for (unsigned int k = 0; k < this->GetNumberOfIterations(); ++k)
  {
    /** Reset the iterators. */
    nit.GoToBegin();
    nit2.GoToBegin();
    oit.GoToBegin();

    /** The actual work. */
    while (!nit.IsAtEnd())
    {
      /** Speed up: do not filter locations where c(x) = 0. */
      if (nit2.GetCenterPixel() < 0.000001)
      {
        /** Just copy input to output. */
        oit.Set(nit.GetCenterPixel());
      }
      else
      {
        /** Initialize the sum to 0. */
        for (j = 0; j < InputImageDimension; ++j)
        {
          sum[j] = 0.0;
        }

        /** Initialize sumc. */
        double sumc = 0.0;

        /** Calculate the weighted mean over the neighborhood.
         * mean = SUthis->m_i{ ci * x_i } / SUthis->m_i{ ci }
         */
        for (i = 0; i < neighborhoodSize; ++i)
        {
          /** Get current pixel in this neighborhood. */
          InputPixelType pix = nit.GetPixel(i);

          /** Get ci-value on current index. */
          ci = nit2.GetPixel(i);

          /** Calculate SUthis->m_i{ ci } and SUthis->m_i{ ci * x_i }. */
          sumc += ci;
          for (j = 0; j < InputImageDimension; ++j)
          {
            sum[j] += ci * static_cast<double>(pix[j]);
          }
        }

        /** Get the mean value by dividing by sumc. */
        InputPixelType mean;
        for (j = 0; j < InputImageDimension; ++j)
        {
          if (sumc < 0.00001)
          {
            mean[j] = 0.0;
          }
          else
          {
            mean[j] = static_cast<ValueType>(sum[j] / sumc);
          }
        }

        /** Get c. */
        c = nit2.GetCenterPixel();

        /** Set 'y = (1 - c) * x + c * mean' to the temporary output. */
        InputPixelType value = nit.GetCenterPixel() * (1.0 - c) + mean * c;

        /** Set it. */
        oit.Set(value);

      } // end if c < 0.000001

      /** Increase all iterators. */
      ++nit;
      ++nit2;
      ++oit;
      // progress.CompletedPixel();

    } // end while

    /** Copy outputtmp to output. */
    if (this->GetNumberOfIterations() > 0)
    {
      out_it.GoToBegin();
      oit.GoToBegin();
      while (!out_it.IsAtEnd())
      {
        out_it.Set(oit.Get());
        ++out_it;
        ++oit;
      }
    } // end if GetNumberOfIterations() > 0

  } // end for NumberOfIterations

} // end GenerateData()


/**
 * ********************* PrintSelf ******************************
 */

template <class TInputImage, class TGrayValueImage>
void
VectorMeanDiffusionImageFilter<TInputImage, TGrayValueImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Radius: " << this->m_Radius << std::endl;

} // end PrintSelf()


/**
 * ******************** SetGrayValueImage ***********************
 */

template <class TInputImage, class TGrayValueImage>
void
VectorMeanDiffusionImageFilter<TInputImage, TGrayValueImage>::SetGrayValueImage(GrayValueImageType * _arg)
{
  if (this->m_GrayValueImage != _arg)
  {
    this->m_GrayValueImage = _arg;
  }

} // end SetGrayValueImage()


/**
 * ******************** FilterGrayValueImage ********************
 *
 * This function reads an image u(x). This image is rescaled to
 * intensities between 0.0 and 1.0, giving u~(x). Then we threshold
 * the coefficient image or not.
 */

template <class TInputImage, class TGrayValueImage>
void
VectorMeanDiffusionImageFilter<TInputImage, TGrayValueImage>::FilterGrayValueImage()
{
  /** This functions rescales the intensities of the input
   * this->m_GrayValueImage between 0 and 1, and converts it to
   * a double image. No thresholding is performed.
   */

  /** Create this->m_Cx. */
  this->m_Cx = DoubleImageType::New();

  /** Rescale intensity of this->m_GrayValueImage to values between
   * 0.0 and 1.0.
   */
  this->m_RescaleFilter = RescaleImageFilterType::New();
  this->m_RescaleFilter->SetOutputMinimum(0.000001);
  this->m_RescaleFilter->SetOutputMaximum(0.999999);
  this->m_RescaleFilter->SetInput(this->m_GrayValueImage);

  /** First set this->m_Cx = rescaleFilter->GetOutput(). */
  this->m_Cx = this->m_RescaleFilter->GetOutput();
  try
  {
    this->m_Cx->Update();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception and throw again. */
    excp.SetLocation("VectorMeanDiffusionImageFilter - FilterGrayValueImage()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while rescaling the intensities of the grayValue image.\n";
    excp.SetDescription(err_str);
    throw;
  }

} // end FilterGrayValueImage()


} // end namespace itk

#endif // end #ifndef itkVectorMeanDiffusionImageFilter_hxx
