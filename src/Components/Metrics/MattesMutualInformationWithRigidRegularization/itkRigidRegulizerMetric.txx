#ifndef __itkRigidRegulizerMetric_cxx
#define __itkRigidRegulizerMetric_cxx

#include "itkRigidRegulizerMetric.h"

#include "itkImageRegionIterator.h"
/** Include splitter and combiner of vector images. */
//#include "itkVectorIndexSelectionCastImageFilter.h"
#include "itkScalarToArrayCastImageFilter.h"

// tmp
#include "itkImageFileWriter.h"

namespace itk
{

  /**
   * ****************** Constructor *******************************
   */
	template< unsigned int Dimension, class TScalarType >
  RigidRegulizerMetric< Dimension, TScalarType >
		::RigidRegulizerMetric()
  {
		/** Initialize member variables. */
		this->m_BSplineTransform = 0;
		this->m_RigidityImage = 0;
		m_UseImageSpacing = true;
		m_SecondOrderWeight = NumericTraits<ScalarType>::One;

  } // end constructor


	/**
   * ****************** GetNumberOfParameters *********************
   */
	template< unsigned int Dimension, class TScalarType >
		unsigned int RigidRegulizerMetric< Dimension, TScalarType >
		::GetNumberOfParameters(void) const
  {
		return this->m_BSplineTransform->GetNumberOfParameters();

	} // end GetNumberOfParameters


	/**
   * *********************** GetValue *****************************
   */
	template< unsigned int Dimension, class TScalarType >
		typename RigidRegulizerMetric< Dimension, TScalarType >
		::MeasureType
		RigidRegulizerMetric< Dimension, TScalarType >
		::GetValue( const ParametersType & parameters ) const
  {
		/** Set the parameters in the transform.
		 * In this function, also the coefficient images are created.
		 */
		this->m_BSplineTransform->SetParameters( parameters );

		/** Calculate the rigid penalty term value. */
		MeasureType value = 0.0;

		/** Return a value. */
		return value;

  } // end GetValue

	
	/**
   * *********************** GetDerivative ************************
   */
	template< unsigned int Dimension, class TScalarType >
	void RigidRegulizerMetric< Dimension, TScalarType >
		::GetDerivative( const ParametersType & parameters, DerivativeType & derivative ) const
  {
		// temp with todo
		typedef ScalarToArrayCastImageFilter<
			CoefficientImageType, CoefficientVectorImageType >		ScalarImageCombineType;
//		typedef typename ScalarImageCombineType::Pointer		ScalarImageCombinePointer;

		/** Set the parameters in the transform.
		 * In this function, also the dimensions of the parameters-array
		 * are checked and the coefficient images are created.
		 */
		this->m_BSplineTransform->SetParameters( parameters );

		/** Get a handle to the coefficient image. */
		CoefficientImagePointer * coefImage;
		coefImage = this->m_BSplineTransform->GetCoefficientImage();

		/** Combine the coefficient images. */
		typename ScalarImageCombineType::Pointer combiner = ScalarImageCombineType::New();
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			combiner->SetInput( i, coefImage[ i ] );
		}
		CoefficientVectorImagePointer		coefVectorImage;
		coefVectorImage = combiner->GetOutput();
		coefVectorImage->Update();

		// tmp write
		typedef ImageFileWriter< CoefficientVectorImageType > WriterType;
		typename WriterType::Pointer writer = WriterType::New();
		writer->SetFileName( "coefImage.mhd" );
		writer->SetInput( coefVectorImage );
		writer->Update();

		/** Create the RigidDerivative filter and image. */
		RigidDerivativeFilterPointer rigidDerivativeFilter = RigidDerivativeFilterType::New();
		CoefficientVectorImagePointer rigidDerivativeImage = CoefficientVectorImageType::New();

		/** Set stuff into the filter. */
		rigidDerivativeFilter->SetUseImageSpacing( this->m_UseImageSpacing );
		rigidDerivativeFilter->SetSecondOrderWeight( this->m_SecondOrderWeight );
		rigidDerivativeFilter->SetRigidityImage( this->m_RigidityImage );
		rigidDerivativeFilter->SetOutputDirectoryName( this->m_OutputDirectoryName.c_str() );

    /** Set the pipeline. */
		// \todo Let the derivativeFilter  accept an array of coefficient images,
		// so that the combining and splitting can be skipped.
		//derivativeFilter->SetInput( this->m_BSplineTransform->GetCoefficientImage() );
		rigidDerivativeFilter->SetInput( coefVectorImage );
		rigidDerivativeImage = rigidDerivativeFilter->GetOutput();

		/** Execute the pipeline. */
		rigidDerivativeImage->Update();

		/** Create the derivative in parameter-form from the derivativeImage. */
		typedef ImageRegionIterator< CoefficientVectorImageType >	IteratorType;
		IteratorType it( rigidDerivativeImage, rigidDerivativeImage->GetLargestPossibleRegion() );
		unsigned int j = 0;
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			it.GoToBegin();
			while ( !it.IsAtEnd() )
			{
				derivative[ j ] = it.Get()[ i ];
				++it;
				j++;
			}
		} // end while
	
  } // end GetDerivative

	
	/**
   * *********************** GetValueAndDerivative ****************
   */
	template< unsigned int Dimension, class TScalarType >
	void RigidRegulizerMetric< Dimension, TScalarType >
		::GetValueAndDerivative( const ParametersType & parameters,
      MeasureType & value, DerivativeType & derivative ) const
  {
		/** Set output values to zero. */
		value = NumericTraits< MeasureType >::Zero;
		derivative = DerivativeType( this->GetNumberOfParameters() );
		derivative.Fill( NumericTraits< MeasureType >::Zero );

		/** Calculate the rigid penalty term value. *
		value = this->GetValue( parameters );

		/** Calculate the derivative of this penalty term. *
		this->GetDerivative( parameters, derivative );*/
	
	//\todo seperate value and derivative.

	// temp with todo
	typedef ScalarToArrayCastImageFilter<
	CoefficientImageType, CoefficientVectorImageType >		ScalarImageCombineType;

		/** Set the parameters in the transform.
		 * In this function, also the dimensions of the parameters-array
		 * are checked and the coefficient images are created.
		 */
		this->m_BSplineTransform->SetParameters( parameters );

		/** Get a handle to the coefficient image. */
		CoefficientImagePointer * coefImage;
		coefImage = this->m_BSplineTransform->GetCoefficientImage();

		/** Combine the coefficient images. */
		typename ScalarImageCombineType::Pointer combiner = ScalarImageCombineType::New();
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			combiner->SetInput( i, coefImage[ i ] );
		}
		CoefficientVectorImagePointer	coefVectorImage;
		coefVectorImage = combiner->GetOutput();
		coefVectorImage->Update();

		// tmp write
		/*typedef ImageFileWriter< CoefficientVectorImageType > WriterType;
		typename WriterType::Pointer writer = WriterType::New();
		writer->SetFileName( "coefImage.mhd" );
		writer->SetInput( coefVectorImage );
		writer->Update();*/

		/** Create the RigidDerivative filter and image. */
		RigidDerivativeFilterPointer	rigidDerivativeFilter = RigidDerivativeFilterType::New();
		CoefficientVectorImagePointer	rigidDerivativeImage = CoefficientVectorImageType::New();

		/** Set stuff into the filter. */
		rigidDerivativeFilter->SetUseImageSpacing( this->m_UseImageSpacing );
		rigidDerivativeFilter->SetSecondOrderWeight( this->m_SecondOrderWeight );
		rigidDerivativeFilter->SetRigidityImage( this->m_RigidityImage );
		rigidDerivativeFilter->SetOutputDirectoryName( this->m_OutputDirectoryName.c_str() );

    /** Set the pipeline. */
		// \todo Let the derivativeFilter  accept an array of coefficient images,
		// so that the combining and splitting can be skipped.
		//derivativeFilter->SetInput( this->m_BSplineTransform->GetCoefficientImage() );
		rigidDerivativeFilter->SetInput( coefVectorImage );
		rigidDerivativeImage = rigidDerivativeFilter->GetOutput();

		/** Execute the pipeline. */
		rigidDerivativeImage->Update();

		/** Create the derivative in parameter-form from the derivativeImage. */
		typedef ImageRegionIterator< CoefficientVectorImageType >	IteratorType;
		IteratorType it( rigidDerivativeImage, rigidDerivativeImage->GetLargestPossibleRegion() );
		unsigned int j = 0;
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			it.GoToBegin();
			while ( !it.IsAtEnd() )
			{
				derivative[ j ] = it.Get()[ i ];
				++it;
				j++;
			}
		} // end while

		/** Get the value. */
		value = rigidDerivativeFilter->GetRigidRegulizerValue();

  } // end GetValueAndDerivative

	/**
	 * ********************* PrintSelf ******************************
	 *
	 * Print out internal information about this class.
	 */

	template< unsigned int Dimension, class TScalarType >
		void RigidRegulizerMetric< Dimension, TScalarType >
		::PrintSelf( std::ostream& os, Indent indent ) const
	{
		/** Call the superclass' PrintSelf. */
		Superclass::PrintSelf( os, indent );
		
		/** Add debugging information. */
		os << indent << "SecondOrderWeight: "
			<< this->m_SecondOrderWeight << std::endl;
		os << indent << "UseImageSpacing: "
			<< this->m_UseImageSpacing << std::endl;
		os << indent << "RigidityImage: "
			<< this->m_RigidityImage << std::endl;
		os << indent << "BSplineTransform: "
			<< this->m_BSplineTransform << std::endl;
		os << indent << "OutputDirectoryName: "
			<< this->m_OutputDirectoryName << std::endl;
		
	} // end PrintSelf

} // end namespace itk

#endif // #ifndef __itkRigidRegulizerMetric_cxx

