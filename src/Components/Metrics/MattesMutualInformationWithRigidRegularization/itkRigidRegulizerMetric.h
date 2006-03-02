#ifndef __itkRigidRegulizerMetric_h
#define __itkRigidRegulizerMetric_h

#include "itkSingleValuedCostFunction.h"

#include "itkBSplineDeformableTransform.h"
#include "itkRigidRegularizationDerivativeImageFilter.h"


namespace itk
{
  /** 
   * \class RigidRegulizerMetric
   * \brief A metric that calculates a rigid penalty term.
   *
	 * The rigid penalty term penalizes deviations from a rigid
	 * transformation at regions specified by the so-called rigidity images.
	 *
	 * This metric only works with B-splines as a transformation model.
	 *
	 * References:
	 * [1] "Nonrigid Registration Using a Rigidity Constraint"
	 *		M. Staring, S. Klein and Josien P.W. Pluim
	 *		SPIE Medical Imaging 2006: Image Processing, 2006.
   * 
	 * \sa BSplineTransform
	 * \sa RigidRegularizationDerivativeImageFilter
	 * \sa MattesMutualInformationImageToImageMetricWithRigidRegularization
	 * \sa MattesMutualInformationMetricWithRigidRegularization
   * \ingroup Metrics
   */

	template< unsigned int Dimension, class TScalarType >
  class RigidRegulizerMetric : public SingleValuedCostFunction
  {
  public:

		/** Standard itk stuff. */
    typedef RigidRegulizerMetric						Self;
    typedef SingleValuedCostFunction        Superclass;
    typedef SmartPointer<Self>              Pointer;
    typedef SmartPointer<const Self>        ConstPointer;

		/** Method for creation through the object factory. */
    itkNewMacro( Self );

		/** Run-time type information (and related methods). */
    itkTypeMacro( RigidRegulizerMetric, SingleValuedCostFunction );

		/** Typedef's inherited from the superclass. */
    typedef typename Superclass::MeasureType         MeasureType;
    typedef typename Superclass::DerivativeType      DerivativeType;
    typedef typename Superclass::ParametersType      ParametersType;

		typedef TScalarType		ScalarType;

		/** Define the dimension. */
		itkStaticConstMacro( ImageDimension, unsigned int, Dimension );

		/** Typedef's for BSpline transform. */
		typedef BSplineDeformableTransform< ScalarType,
			Dimension, 3 >																	BSplineTransformType;
		typedef typename BSplineTransformType::Pointer		BSplineTransformPointer;

		/** Typedef's for the coefficient image (which is a scalar image),
		 * for the coefficient vector image (which is a vector image,
		 * containing all components of the coefficient image), and
		 * for the rigidity image (which contains the rigidity coefficients).
		 */
		typedef typename BSplineTransformType::PixelType			CoefficientPixelType;
		typedef typename BSplineTransformType::ImageType			CoefficientImageType;
		typedef typename CoefficientImageType::Pointer				CoefficientImagePointer;
		typedef Vector< CoefficientPixelType,
			itkGetStaticConstMacro( ImageDimension ) >					CoefficientVectorType;
		typedef Image< CoefficientVectorType,
			itkGetStaticConstMacro( ImageDimension ) >					CoefficientVectorImageType;
		typedef typename CoefficientVectorImageType::Pointer	CoefficientVectorImagePointer;
		typedef CoefficientImageType													RigidityImageType;
		typedef CoefficientImagePointer												RigidityImagePointer;

		/** Typedef's for the rigid derivative filter. */
		typedef RigidRegularizationDerivativeImageFilter<
			CoefficientVectorImageType,
			CoefficientVectorImageType >												RigidDerivativeFilterType;
		typedef typename RigidDerivativeFilterType::Pointer		RigidDerivativeFilterPointer;

    /** The GetValue()-method returns the rigid penalty value. */
    virtual MeasureType GetValue(
			const ParametersType & parameters ) const;

    /** The GetDerivative()-method returns the rigid penalty derivative. */
    virtual void GetDerivative(
			const ParametersType & parameters,
			DerivativeType & derivative ) const;

    /** The GetValueAndDerivative()-method returns the rigid penalty value and its derivative. */
    virtual void GetValueAndDerivative(
      const ParametersType & parameters,
      MeasureType & value,
      DerivativeType & derivative ) const;

    /** Get the number of parameters. */
    virtual unsigned int GetNumberOfParameters(void) const;

		/** Set the BSpline transform in this class.
		 * This class expects a BSplineTransform! It is not suited for others.
		 */
		itkSetObjectMacro( BSplineTransform, BSplineTransformType );

		/** Set the RigidityImage in this class. */
		itkSetObjectMacro( RigidityImage, RigidityImageType );

		/** Set macro for using the spacing. */
		itkSetMacro( UseImageSpacing, bool );

		/** Set macro for the weight of the second order term. */
		itkSetMacro( SecondOrderWeight, ScalarType );

		/** Set macro for the weight of the orthonormality term. */
		itkSetMacro( OrthonormalityWeight, ScalarType );

		/** Set macro for the weight of the properness term. */
		itkSetMacro( PropernessWeight, ScalarType );

		/** Set the OutputDirectoryName. */
		itkSetStringMacro( OutputDirectoryName );

  protected:

		/** The constructor. */
    RigidRegulizerMetric();
		/** The destructor. */
    virtual ~RigidRegulizerMetric() {};

		/** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const;

  private:

		/** The private constructor. */
    RigidRegulizerMetric( const Self& );	// purposely not implemented
		/** The private copy constructor. */
    void operator=( const Self& );				// purposely not implemented

		/** Member variables. */
		BSplineTransformPointer		m_BSplineTransform;
		RigidityImagePointer			m_RigidityImage;

		bool				m_UseImageSpacing;
		ScalarType	m_SecondOrderWeight;
		ScalarType	m_OrthonormalityWeight;
		ScalarType	m_PropernessWeight;

		/** Name of the output directory. */
		std::string m_OutputDirectoryName;

  }; // end class RigidRegulizerMetric
  

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRigidRegulizerMetric.txx"
#endif

#endif // #ifndef __itkRigidRegulizerMetric_h

