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
#ifndef __itkNormalizedCorrelationImageToImageMetric2_h
#define __itkNormalizedCorrelationImageToImageMetric2_h

#include "itkImageToImageMetric.h"
#include "itkCovariantVector.h"
#include "itkPoint.h"


namespace itk
{
/** \class NormalizedCorrelationImageToImageMetric2
 * \brief Computes similarity between two images to be registered
 *
 * This metric computes the correlation between pixels in the fixed image
 * and pixels in the moving image. The spatial correspondance between 
 * fixed and moving image is established through a Transform. Pixel values are
 * taken from the fixed image, their positions are mapped to the moving
 * image and result in general in non-grid position on it. Values at these
 * non-grid position of the moving image are interpolated using a user-selected
 * Interpolator. The correlation is normalized by the autocorrelations of both
 * the fixed and moving images.
 *
 * This class provides functionality to calculate (the derivative of) the
 * normalized correlation on only a subset of the fixed image voxels. This
 * option is controlled by the boolean UseAllPixels, which is by default true.
 * Substantial speedup can be accomplished by setting it to false and specifying
 * the NumberOfSpacialSamples to some small portion of the total number of fixed
 * image samples. The samples are randomly chosen using an
 * itk::ImageRandomConstIteratorWithIndex Every iteration a new set of those
 * samples are used. This is important, because the error made by calculating
 * the metric value with only a subset of all samples should be randomly
 * distributed with zero mean.
 *
 * The normalized correlation NC is defined as:
 *
 *                 sum_x[ f(x) * m(x+u(x,p)) ]
 * NC = ------------------------------------------------
 *      sqrt[ sum_x[ f(x)^2 ] * sum_x[ m(x+u(x,p))^2 ] ]
 *
 *             sfm
 *    = ----------------- ,
 *      sqrt[ sff * smm ]
 *
 * where x a voxel in the fixed image f, m the moving image, u(x,p) the
 * deformation of x depending on the transform parameters p. sfm, sff and smm
 * is notation used in the source code. The derivative of NC to p equals:
 * 
 * dNC/dp = dNC/dm * dm/dx * dx/dp = dNC/dm * gradient * jacobian,
 *
 * where gradient is the derivative of the moving image m to x, as calculated
 * in the Initialize of the itk::ImageToImageMetric, and where jacobian is the
 * derivative of the transformation to its parameters, implemented in the
 * GetJacobian() of the transform. gradient * jacobian is called the differential.
 * This yields for the derivative:
 *
 *           sum_x[ f(x) * differential ] - ( sfm / smm ) * sum_x[ m(x+u(x,p)) * differential ]
 * dNC/dp = -----------------------------------------------------------------------------------
 *                                         sqrt[ sff * smm ]
 *
 * This class has an option to subtract the sample mean from the sample values
 * in the cross correlation formula. This typically results in narrower valleys
 * in the cost fucntion NC. The default value is false. If SubtractMean is true,
 * the NC is defined as:
 *
 *                sum_x[ ( f(x) - Af ) * ( m(x+u(x,p)) - Am ) ]
 * NC = ------------------------------------------------------------------
 *      sqrt[ sum_x[ ( f(x) - Af )^2 ] * sum_x[ ( m(x+u(x,p)) - Am )^2 ] ]
 *
 *                       sfm - sf * sm / N
 *    = ----------------------------------------------------- ,
 *      sqrt[ ( sff - sf * sf / N ) * ( smm - sm * sm / N ) ]
 *
 * where Af and Am are the average of f and m, respectively.
 *
 * \todo In the while loop in GetValue and GetValueAndDerivative another for
 * loop is made over all parameters. In case of a B-spline transform advantage
 * can be taken from the fact that it has compact support, similar to the
 * itk::MattesMutualInformationImageToImageMetric.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT NormalizedCorrelationImageToImageMetric2 : 
    public ImageToImageMetric< TFixedImage, TMovingImage >
{
public:

  /** Standard class typedefs. */
  typedef NormalizedCorrelationImageToImageMetric2				Self;
  typedef ImageToImageMetric<TFixedImage, TMovingImage >  Superclass;

  typedef SmartPointer<Self>         Pointer;
  typedef SmartPointer<const Self>   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );
 
  /** Run-time type information (and related methods). */
  itkTypeMacro( NormalizedCorrelationImageToImageMetric2, Object );
 
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

	/** The fixed image dimension. */
	itkStaticConstMacro( FixedImageDimension, unsigned int,
		FixedImageType::ImageDimension );

	/** The moving image dimension. */
	itkStaticConstMacro( MovingImageDimension, unsigned int,
		MovingImageType::ImageDimension );

  /** Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType & parameters ) const;
	MeasureType GetValueUsingAllPixels( const TransformParametersType & parameters ) const;
	MeasureType GetValueUsingSomePixels( const TransformParametersType & parameters ) const;

	/** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters,
		DerivativeType & Derivative ) const;

  /** Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
		MeasureType& Value, DerivativeType& Derivative ) const;
	void GetValueAndDerivativeUsingAllPixels( const TransformParametersType & parameters,
		MeasureType& Value, DerivativeType& Derivative ) const;
	void GetValueAndDerivativeUsingSomePixels( const TransformParametersType & parameters,
		MeasureType& Value, DerivativeType& Derivative ) const;

  /** Set/Get SubtractMean boolean. If true, the sample mean is subtracted 
   * from the sample values in the cross-correlation formula and
   * typically results in narrower valleys in the cost fucntion.
   * Default value is false. */
  itkSetMacro( SubtractMean, bool );
  itkGetConstReferenceMacro( SubtractMean, bool );
  itkBooleanMacro( SubtractMean );

	/** Set/Get UseAllPixels boolean. If false, a random set of samples
	 * is used for calculating the value and derivative. If true, all
	 * pixels are used for this. Default value is true. */
	itkSetMacro( UseAllPixels, bool );
  itkGetConstReferenceMacro( UseAllPixels, bool );
  itkBooleanMacro( UseAllPixels );

	/** Set/Get NumberOfSpatialSamples long. This number defines how much
	 * samples are used to calculate the value and derivative. This number
	 * is only relevant if UseAllPixels is false. */
	itkSetMacro( NumberOfSpatialSamples, unsigned long );
	itkGetConstReferenceMacro( NumberOfSpatialSamples, unsigned long );

protected:
  NormalizedCorrelationImageToImageMetric2();
  virtual ~NormalizedCorrelationImageToImageMetric2() {};
  void PrintSelf( std::ostream& os, Indent indent ) const;

private:
  NormalizedCorrelationImageToImageMetric2(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool    m_SubtractMean;
	bool		m_UseAllPixels;
	unsigned long		m_NumberOfSpatialSamples;

}; // end class NormalizedCorrelationImageToImageMetric2

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkNormalizedCorrelationImageToImageMetric2.txx"
#endif

#endif // end #ifndef __itkNormalizedCorrelationImageToImageMetric2_h

