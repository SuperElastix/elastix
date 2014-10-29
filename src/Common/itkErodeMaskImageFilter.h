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
#ifndef __itkErodeMaskImageFilter_h
#define __itkErodeMaskImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkMultiResolutionPyramidImageFilter.h"

namespace itk
{
/**
 * \class ErodeMaskImageFilter
 *
 * This filter computes the Erosion of a mask image.
 * It makes only sense for masks used in a multiresolution registration procedure.
 *
 * The input to this filter is a scalar-valued itk::Image of arbitrary
 * dimension. The output is a scalar-valued itk::Image, of the same type
 * as the input image. This restriction is not really necessary,
 * but easier for coding.
 *
 * If IsMovingMask == false:\n
 *   If more resolution levels are used, the image is subsampled. Before
 *   subsampling the image is smoothed with a Gaussian filter, with variance
 *   (schedule/2)^2. The 'schedule' depends on the resolution level.
 *   The 'radius' of the convolution filter is roughly twice the standard deviation.
 *   Thus, the parts in the edge with size 'radius' are influenced by the background.\n
 *   --> <tt>radius = static_cast<unsigned long>( schedule + 1 );</tt>
 *
 * If IsMovingMask == true:\n
 *   Same story as before. Now the size the of the eroding element is doubled.
 *   This is because the gradient of the moving image is used for calculating
 *   the derivative of the metric.\n
 *   --> <tt>radius = static_cast<unsigned long>( 2 * schedule + 1 );</tt>
 *
 *
 * \sa ParabolicErodeImageFilter
 *
 **/

template< class TImage >
class ErodeMaskImageFilter :
  public ImageToImageFilter< TImage, TImage >
{
public:

  /** Standard ITK stuff. */
  typedef ErodeMaskImageFilter                 Self;
  typedef ImageToImageFilter< TImage, TImage > Superclass;
  typedef SmartPointer< Self >                 Pointer;
  typedef SmartPointer< const Self >           ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( ErodeMaskImageFilter, ImageToImageFilter );

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Typedefs. */
  typedef TImage                              InputImageType;
  typedef TImage                              OutputImageType;
  typedef typename InputImageType::Pointer    InputImagePointer;
  typedef typename OutputImageType::Pointer   OutputImagePointer;
  typedef typename InputImageType::PixelType  InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;

  /** Dimensionality of the two images is assumed to be the same. */
  itkStaticConstMacro( InputImageDimension, unsigned int,
    InputImageType::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int,
    OutputImageType::ImageDimension );
  itkStaticConstMacro( ImageDimension, unsigned int,
    OutputImageType::ImageDimension );

  /** Define the schedule type. */
  typedef MultiResolutionPyramidImageFilter<
    InputImageType, OutputImageType >                    ImagePyramidFilterType;
  typedef typename ImagePyramidFilterType::ScheduleType ScheduleType;

  /** Set/Get the pyramid schedule used to downsample the image whose
   * mask is the input of the ErodeMaskImageFilter
   * Default: filled with ones, one resolution.
   */
  virtual void SetSchedule( const ScheduleType & schedule )
  {
    this->m_Schedule = schedule;
    this->Modified();
  }


  itkGetConstReferenceMacro( Schedule, ScheduleType );

  /** Set/Get whether the mask serves as a 'moving mask' in the registration
   * Moving masks are eroded with a slightly larger kernel, because the
   * derivative is usually taken on the moving image.
   * Default: false
   */
  itkSetMacro( IsMovingMask, bool );
  itkGetConstMacro( IsMovingMask, bool );

  /** Set the resolution level of the registration. Default: 0. */
  itkSetMacro( ResolutionLevel, unsigned int );
  itkGetConstMacro( ResolutionLevel, unsigned int );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro( SameDimensionCheck,
    ( Concept::SameDimension< InputImageDimension, OutputImageDimension > ) );
  /** End concept checking */
#endif

protected:

  /** Constructor. */
  ErodeMaskImageFilter();

  /** Destructor */
  virtual ~ErodeMaskImageFilter(){}

  /** Standard pipeline method. While this class does not implement a
   * ThreadedGenerateData(), its GenerateData() delegates all
   * calculations to the ParabolicErodeImageFilter, which is multi-threaded.
   */
  virtual void GenerateData( void );

private:

  ErodeMaskImageFilter( const Self & );    // purposely not implemented
  void operator=( const Self & );          // purposely not implemented

  bool         m_IsMovingMask;
  unsigned int m_ResolutionLevel;
  ScheduleType m_Schedule;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkErodeMaskImageFilter.hxx"
#endif

#endif
