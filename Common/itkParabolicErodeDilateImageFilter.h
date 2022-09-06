/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkParabolicErodeDilateImageFilter_h
#define itkParabolicErodeDilateImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkNumericTraits.h"
#include "itkProgressReporter.h"

namespace itk
{
/**
 * \class ParabolicErodeDilateImageFilter
 * \brief Parent class for morphological operations with parabolic
 * structuring elements.
 *
 * Parabolic structuring elements are the morphological counterpart of
 * the gaussian function in linear processing. Parabolic structuring
 * elements are dimensionally decomposable and fast algorithms are
 * available for computing erosions and dilations along lines.
 * This class implements the "point of contact" algorithm, which is
 * reasonably efficient. Improved versions, that are nearly
 * independent of structuring function size, are also known but
 * haven't been implemented here.
 *
 * Parabolic structuring functions can be used as a fast alternative
 * to the "rolling ball" structuring element classically used in
 * background estimation, for example in ImageJ, have applications in
 * image sharpening and distance transform computation.
 *
 * This class uses an internal buffer of RealType pixels for each
 * line. This line is cast to the output pixel type when written back
 * to the output image. Since the filter uses dimensional
 * decomposition this approach could result in inaccuracy as pixels
 * are cast back and forth between low and high precision types. Use a
 * high precision output type and cast manually if this is a problem.
 *
 * This filter is threaded. Threading mechanism derived from
 * SignedMaurerDistanceMap extensions by Gaetan Lehman
 *
 * \author Richard Beare, Department of Medicine, Monash University,
 * Australia.  <Richard.Beare@med.monash.edu.au>
 *
 **/

template <typename TInputImage, bool doDilate, typename TOutputImage = TInputImage>
class ITK_TEMPLATE_EXPORT ParabolicErodeDilateImageFilter : public ImageToImageFilter<TInputImage, TOutputImage>
{

public:
  ITK_DISALLOW_COPY_AND_MOVE(ParabolicErodeDilateImageFilter);

  /** Standard class typedefs. */
  using Self = ParabolicErodeDilateImageFilter;
  using Superclass = ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ParabolicErodeDilateImageFilter, ImageToImageFilter);

  /** Pixel Type of the input image */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using PixelType = typename TInputImage::PixelType;
  using RealType = typename NumericTraits<PixelType>::RealType;
  using ScalarRealType = typename NumericTraits<PixelType>::ScalarRealType;
  using OutputPixelType = typename TOutputImage::PixelType;

  /** Smart pointer typedef support.  */
  using InputImagePointer = typename TInputImage::Pointer;
  using InputImageConstPointer = typename TInputImage::ConstPointer;
  using InputSizeType = typename TInputImage::SizeType;
  using OutputSizeType = typename TOutputImage::SizeType;

  /** a type to represent the "kernel radius" */
  using RadiusType = typename itk::FixedArray<ScalarRealType, TInputImage::ImageDimension>;

  /** Image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);

  using OutputImageRegionType = typename OutputImageType::RegionType;
  /** Define the image type for internal computations
      RealType is usually 'double' in NumericTraits.
      Here we prefer float in order to save memory.  */

  using InternalRealType = typename NumericTraits<PixelType>::FloatType;
  // typedef typename Image<InternalRealType, Self::ImageDimension >   RealImageType;

  // set all of the scales the same
  void
  SetScale(ScalarRealType scale);

  itkSetMacro(Scale, RadiusType);
  itkGetConstReferenceMacro(Scale, RadiusType);

  /**
   * Set/Get whether the scale refers to pixels or world units -
   * default is false
   */
  itkSetMacro(UseImageSpacing, bool);
  itkGetConstReferenceMacro(UseImageSpacing, bool);
  itkBooleanMacro(UseImageSpacing);
  /** Image related typedefs. */

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(SameDimension, (Concept::SameDimension<Self::InputImageDimension, Self::OutputImageDimension>));

  itkConceptMacro(Comparable, (Concept::Comparable<PixelType>));

  /** End concept checking */
#endif

protected:
  ParabolicErodeDilateImageFilter();
  ~ParabolicErodeDilateImageFilter() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Generate Data */
  void
  GenerateData() override;

  int
  SplitRequestedRegion(int i, int num, OutputImageRegionType & splitRegion);

  void
  ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread, ThreadIdType threadId) override;

  //  virtual void GenerateInputRequestedRegion();
  // Override since the filter produces the entire dataset.
  void
  EnlargeOutputRequestedRegion(DataObject * output) override;

  bool m_UseImageSpacing;

private:
  RadiusType                      m_Scale;
  typename TInputImage::PixelType m_Extreme;

  int m_MagnitudeSign;
  int m_CurrentDimension;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkParabolicErodeDilateImageFilter.hxx"
#endif

#endif
