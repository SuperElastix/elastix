/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkErodeMaskImageFilter_h
#define __itkErodeMaskImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"
#include "itkArray2D.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkMultiResolutionPyramidImageFilter.h"

namespace itk
{
/**
 * \class ErodeMaskImageFilter
 *
 * This filter computes the Erosion of a mask image. 
 * It makes only sense for masks used in a 
 * multiresolution registration procedure.
 * 
 * The input to this filter is a scalar-valued itk::Image of arbitrary
 * dimension. The output is a scalar-valued itk::Image, of the same type
 * as the input image. This restriction is not really necessary,
 * but easier for coding ;-).
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
 * \sa BinaryErodeImageFilter
 *
 **/

template <class TImage>
class ErodeMaskImageFilter : 
    public ImageToImageFilter< TImage, TImage > 
{
public:
  /** Standard "Self" & Superclass typedef.   */
  typedef ErodeMaskImageFilter Self;
  typedef ImageToImageFilter< TImage, TImage > Superclass;
  
  /** Smart pointer typedef support.   */  
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self>  ConstPointer;
  
  /** Run-time type information (and related methods)  */
  itkTypeMacro(ErodeMaskImageFilter, ImageToImageFilter);
  
  /** Method for creation through the object factory.  */
  itkNewMacro(Self);

  /** Typedefs. */
  typedef TImage                                InputImageType;
  typedef TImage                                OutputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename InputImageType::PixelType    InputPixelType;
  typedef typename OutputImageType::PixelType   OutputPixelType;

  /** Dimensionality of the two images is assumed to be the same. */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      InputImageType::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      OutputImageType::ImageDimension);
  itkStaticConstMacro(ImageDimension, unsigned int,
                      OutputImageType::ImageDimension);

  /** Define the schedule type */
  typedef MultiResolutionPyramidImageFilter<
    InputImageType, OutputImageType>           ImagePyramidFilterType;
  typedef typename ImagePyramidFilterType::ScheduleType  ScheduleType;

  /** Set/Get the pyramid schedule used to downsample the image whose
   * mask is the input of the ErodeMaskImageFilter 
   * Default: filled with ones, one resolution  */
  virtual void SetSchedule( const ScheduleType & schedule )
  {
    this->m_Schedule = schedule;
    this->Modified();
  }
  itkGetConstReferenceMacro( Schedule, ScheduleType );

  /** Set/Get whether the mask serves as a 'moving mask' in the registration
   * Moving masks are eroded with a slightly larger kernel, because the derivative
   * is usually taken on the moving image
   * Default: false */
  itkSetMacro( IsMovingMask, bool );
  itkGetConstMacro( IsMovingMask, bool );

  /** Set the resolution level of the registration.
   * Default: 0; */
  itkSetMacro( ResolutionLevel, unsigned int );
  itkGetConstMacro( ResolutionLevel, unsigned int );
  

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(SameDimensionCheck,
    (Concept::SameDimension<InputImageDimension, OutputImageDimension>));
  /** End concept checking */
#endif

protected:

  /** Constructor. */
  ErodeMaskImageFilter();

  /** Destructor */
  virtual ~ErodeMaskImageFilter(){}

  /** Some typedef's used for eroding the masks*/
  typedef BinaryBallStructuringElement<
    InputPixelType, 
    itkGetStaticConstMacro(InputImageDimension) >         StructuringElementType;
  typedef typename StructuringElementType::RadiusType     RadiusType;
  typedef BinaryErodeImageFilter<
    InputImageType,
    OutputImageType,
    StructuringElementType >                              ErodeFilterType;
  typedef typename ErodeFilterType::Pointer               ErodeFilterPointer;
  typedef itk::FixedArray<
    ErodeFilterPointer,
    itkGetStaticConstMacro(InputImageDimension) >             ErodeFilterArrayType;
  

  ErodeFilterArrayType     m_ErodeFilterArray;

  /** Standard pipeline method. While this class does not implement a
   * ThreadedGenerateData(), its GenerateData() delegates all
   * calculations to an BinaryErodeImageFilter.  Since the
   * BinaryErodeImageFilter is multithreaded, this filter is
   * multithreaded by default.   */
  virtual void GenerateData();


private:
  ErodeMaskImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool m_IsMovingMask;
  unsigned int m_ResolutionLevel;
  ScheduleType m_Schedule;

  
};
  
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkErodeMaskImageFilter.txx"
#endif

#endif
