/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkMutualInformationHistogramImageToImageMetricWithMask_h
#define __itkMutualInformationHistogramImageToImageMetricWithMask_h

#include "itkHistogramImageToImageMetric.h"

/** Added for the support of masks.*/
#include "itkMaskImage.h"
//TODO: add support for masks!

namespace itk
{
  /** \class MutualInformationHistogramImageToImageMetricWithMask
      \brief Computes the mutual information between two images to
      be registered using the histograms of the intensities in the images.

      This class is templated over the type of the fixed and moving
      images to be compared.

      This metric computes the similarity measure between pixels in the
      moving image and pixels in the fixed images using a histogram.

      \ingroup RegistrationMetrics */
template <class TFixedImage, class TMovingImage>
class ITK_EXPORT MutualInformationHistogramImageToImageMetricWithMask :
public HistogramImageToImageMetric<TFixedImage, TMovingImage>
{
 public:

  /** Standard class typedefs. */
  typedef MutualInformationHistogramImageToImageMetricWithMask		Self;
  typedef HistogramImageToImageMetric<TFixedImage, TMovingImage>	Superclass;
  typedef SmartPointer<Self>																			Pointer;
  typedef SmartPointer<const Self>																ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MutualInformationHistogramImageToImageMetricWithMask,
    HistogramImageToImageMetric );

  /** Types transferred from the base class */
  typedef typename Superclass::RealType                 RealType;
  typedef typename Superclass::TransformType            TransformType;
  typedef typename Superclass::TransformPointer         TransformPointer;
  typedef typename Superclass::TransformParametersType	TransformParametersType;
  typedef typename Superclass::TransformJacobianType    TransformJacobianType;
  typedef typename Superclass::GradientPixelType        GradientPixelType;

  typedef typename Superclass::MeasureType              MeasureType;
  typedef typename Superclass::DerivativeType           DerivativeType;
  typedef typename Superclass::FixedImageType           FixedImageType;
  typedef typename Superclass::MovingImageType          MovingImageType;
  typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer	MovingImageConstPointer;

  typedef typename Superclass::HistogramType            HistogramType;
  typedef typename HistogramType::FrequencyType         HistogramFrequencyType;
  typedef typename HistogramType::Iterator              HistogramIteratorType;
  typedef typename HistogramType::MeasurementVectorType	HistogramMeasurementVectorType;

	//
	typedef typename Superclass::ScalesType								ScalesType;

	/** Index and Point typedef support. */
  typedef typename FixedImageType::IndexType            FixedImageIndexType;
  typedef typename FixedImageIndexType::IndexValueType  FixedImageIndexValueType;
  typedef typename MovingImageType::IndexType           MovingImageIndexType;
  typedef typename TransformType::InputPointType        FixedImagePointType;
  typedef typename TransformType::OutputPointType       MovingImagePointType;

	/** Typedefs added for masks.*/
	typedef char																					MaskPixelType;
	typedef typename FixedImagePointType::CoordRepType		FixedCoordRepType;
	typedef typename MovingImagePointType::CoordRepType		MovingCoordRepType;
	typedef MaskImage<MaskPixelType,
		::itk::GetImageDimension<FixedImageType>::ImageDimension,
		FixedCoordRepType>																	FixedMaskImageType;
	typedef MaskImage<MaskPixelType,
		::itk::GetImageDimension<MovingImageType>::ImageDimension,
		MovingCoordRepType>																	MovingMaskImageType;		
	typedef typename FixedMaskImageType::Pointer					FixedMaskImagePointer;
	typedef typename MovingMaskImageType::Pointer					MovingMaskImagePointer;
	
	/** Set and Get the masks.*/
	itkSetObjectMacro( FixedMask, FixedMaskImageType );
	itkSetObjectMacro( MovingMask, MovingMaskImageType );
	itkGetObjectMacro( FixedMask, FixedMaskImageType );
	itkGetObjectMacro( MovingMask, MovingMaskImageType );

protected:

  /** Constructor is protected to ensure that \c New() function is used to
      create instances. */
  MutualInformationHistogramImageToImageMetricWithMask();
  virtual ~MutualInformationHistogramImageToImageMetricWithMask(){}

  /** Evaluates the mutual information from the histogram. */
  virtual MeasureType EvaluateMeasure(HistogramType& histogram) const;

	/** Masks.*/
	FixedMaskImagePointer		m_FixedMask;
	MovingMaskImagePointer	m_MovingMask;

private:

  // Purposely not implemented.
  MutualInformationHistogramImageToImageMetricWithMask(Self const&);
  void operator=(Self const&); // Purposely not implemented.

}; // end class MutualInformationHistogramImageToImageMetricWithMask

} // End namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMutualInformationHistogramImageToImageMetricWithMask.txx"
#endif

#endif // __itkMutualInformationHistogramImageToImageMetricWithMask_h
