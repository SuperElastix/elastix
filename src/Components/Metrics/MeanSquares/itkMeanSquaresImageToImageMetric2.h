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
#ifndef __itkMeanSquaresImageToImageMetric2_h
#define __itkMeanSquaresImageToImageMetric2_h

#include "itkImageToImageMetricWithSampling.h"
#include "itkCovariantVector.h"
#include "itkPoint.h"


namespace itk
{
/** \class MeanSquaresImageToImageMetric2
 * \brief Computes similarity between two objects to be registered
 *
 * This Class is templated over the type of the fixed and moving
 * images to be compared.
 *
 * This metric computes the sum of squared differenced between pixels in
 * the moving image and pixels in the fixed image. The spatial correspondance 
 * between both images is established through a Transform. Pixel values are
 * taken from the Moving image. Their positions are mapped to the Fixed image
 * and result in general in non-grid position on it. Values at these non-grid
 * position of the Fixed image are interpolated using a user-selected Interpolator.
 *
 * This class provides functionality to calculate (the derivative of) the
 * mean squares metric on only a subset of the fixed image voxels. This
 * option is controlled by the boolean UseAllPixels, which is by default true.
 * Substantial speedup can be accomplished by setting it to false and specifying
 * the NumberOfSpacialSamples to some small portion of the total number of fixed
 * image samples. The samples are randomly chosen using an
 * itk::ImageRandomConstIteratorWithIndex Every iteration a new set of those
 * samples are used. This is important, because the error made by calculating
 * the metric value with only a subset of all samples should be randomly
 * distributed with zero mean.
 *
 * \todo In the while loop in GetValue and GetValueAndDerivative another for
 * loop is made over all parameters. In case of a B-spline transform advantage
 * can be taken from the fact that it has compact support, similar to the
 * itk::MattesMutualInformationImageToImageMetric.
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT MeanSquaresImageToImageMetric2 : 
    public ImageToImageMetricWithSampling< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef MeanSquaresImageToImageMetric2		Self;
  typedef ImageToImageMetricWithSampling<
    TFixedImage, TMovingImage >             Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );
 
  /** Run-time type information (and related methods). */
  itkTypeMacro( MeanSquaresImageToImageMetric2, ImageToImageMetricWithSampling );

  /** Types transferred from the base class */
  typedef typename Superclass::RealType                 RealType;
  typedef typename Superclass::TransformType            TransformType;
  typedef typename Superclass::TransformPointer         TransformPointer;
  typedef typename Superclass::TransformParametersType  TransformParametersType;
  typedef typename Superclass::TransformJacobianType    TransformJacobianType;
  typedef typename Superclass::GradientPixelType        GradientPixelType;
  typedef typename Superclass::MeasureType              MeasureType;
  typedef typename Superclass::DerivativeType           DerivativeType;
  typedef typename Superclass::FixedImageType           FixedImageType;
  typedef typename Superclass::MovingImageType          MovingImageType;
  typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer  MovingImageConstPointer;
  typedef typename Superclass::InputPointType			      InputPointType;
  typedef typename Superclass::OutputPointType		      OutputPointType;
  typedef typename Superclass::ImageSamplerType         ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer      ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType      ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer   ImageSampleContainerPointer;

	/** The fixed image dimension. */
	itkStaticConstMacro( FixedImageDimension, unsigned int,
		FixedImageType::ImageDimension );

	/** The moving image dimension. */
	itkStaticConstMacro( MovingImageDimension, unsigned int,
		MovingImageType::ImageDimension );

	/** Get the value for single valued optimizers. */
	MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters,
    DerivativeType & derivative ) const;

  /** Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
		MeasureType& Value, DerivativeType& Derivative ) const;

protected:
  MeanSquaresImageToImageMetric2();
  virtual ~MeanSquaresImageToImageMetric2() {};
	void PrintSelf( std::ostream& os, Indent indent ) const;

private:
  MeanSquaresImageToImageMetric2(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

}; // end class MeanSquaresImageToImageMetric2

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMeanSquaresImageToImageMetric2.txx"
#endif

#endif // end #ifndef __itkMeanSquaresImageToImageMetric2_h

