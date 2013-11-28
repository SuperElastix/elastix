/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkMultiResolutionShrinkPyramidImageFilter_h
#define __itkMultiResolutionShrinkPyramidImageFilter_h

#include "itkMultiResolutionPyramidImageFilter.h"


namespace itk
{

/** \class MultiResolutionShrinkPyramidImageFilter
 * \brief Framework for creating images in a multi-resolution
 * pyramid.
 *
 * MultiResolutionShrinkPyramidImageFilter simply shrinks the input images.
 * No smoothing or any other operation is performed. This is useful for
 * example for registering binary images.
 *
 * \sa ShrinkImageFilter
 *
 * \ingroup PyramidImageFilter Multithreaded Streamed
 */
template <
  class TInputImage,
  class TOutputImage
  >
class MultiResolutionShrinkPyramidImageFilter :
    public MultiResolutionPyramidImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef MultiResolutionShrinkPyramidImageFilter             Self;
  typedef MultiResolutionPyramidImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MultiResolutionShrinkPyramidImageFilter,
    MultiResolutionPyramidImageFilter );

  /** ImageDimension enumeration. */
  itkStaticConstMacro( ImageDimension, unsigned int,
    TInputImage::ImageDimension );
  itkStaticConstMacro( OutputImageDimension, unsigned int,
    TOutputImage::ImageDimension );

  /** Inherit types from Superclass. */
  typedef typename Superclass::ScheduleType           ScheduleType;
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::InputImageConstPointer InputImageConstPointer;

  /** Overwrite the Superclass implementation: no padding required. */
  virtual void GenerateInputRequestedRegion( void );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(SameDimensionCheck,
    (Concept::SameDimension<ImageDimension, OutputImageDimension>));
  itkConceptMacro(OutputHasNumericTraitsCheck,
    (Concept::HasNumericTraits<typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

protected:
  MultiResolutionShrinkPyramidImageFilter() {};
  ~MultiResolutionShrinkPyramidImageFilter() {};

  /** Generate the output data. */
  virtual void GenerateData( void );

private:
  MultiResolutionShrinkPyramidImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};


} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiResolutionShrinkPyramidImageFilter.hxx"
#endif

#endif
