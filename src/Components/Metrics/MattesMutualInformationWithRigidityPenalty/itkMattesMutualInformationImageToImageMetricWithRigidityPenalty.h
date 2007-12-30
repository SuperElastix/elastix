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
#ifndef __itkMattesMutualInformationImageToImageMetricWithRigidityPenalty_H__
#define __itkMattesMutualInformationImageToImageMetricWithRigidityPenalty_H__

/** The metric we inherit from. */
#include "itkCombinedImageToImageMetric.h"

/** Include the two metrics we want to combine. */
#include "../MattesMutualInformation/itkMattesMutualInformationImageToImageMetric2.h"
#include "itkRigidityPenaltyTermMetric.h"

/** Include stuff needed for the construction of the rigidity coefficient image. */
#include "itkGrayscaleDilateImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkImageRegionIterator.h"

namespace itk
{
	
	/**
	 * \class MattesMutualInformationImageToImageMetricWithRigidityPenalty
	 * \brief Computes the mutual information between two images to be
	 * registered using the method of Mattes et al. and adds a rigidity penalty term.
	 *
   * \todo: use ParzenWindowMutualInformationImageToImageMetric instead of 
   * MattesMutualInformationImageToImageMetric
   * \todo: incorporate regularisation terms in a more generic way.
   * 
	 * MattesMutualInformationImageToImageMetricWithRigidityPenalty computes the mutual 
	 * information between a fixed and moving image to be registered and adds a
	 * rigidity penalty term. The rigidity penalty term penalizes deviations from a rigid
	 * transformation at regions specified by the so-called rigidity coefficient images.
   *
	 * This metric only works with B-splines as a transformation model.
 	 *
	 * \sa MattesMutualInformationImageToImageMetric2
	 * \sa MattesMutualInformationMetricWithRigidityPenalty
	 * \sa RigidRegulizerMetric
	 * \sa BSplineTransform
	 * \ingroup Metrics
	 */

	template < class TFixedImage, class TMovingImage >
		class MattesMutualInformationImageToImageMetricWithRigidityPenalty :
	public CombinedImageToImageMetric< TFixedImage, TMovingImage >
	{

	public:
		
		/** Standard class typedefs. */
		typedef MattesMutualInformationImageToImageMetricWithRigidityPenalty		Self;
		typedef CombinedImageToImageMetric<
			TFixedImage, TMovingImage >													Superclass;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MattesMutualInformationImageToImageMetricWithRigidityPenalty,
			CombinedImageToImageMetric );
		
		/** Types inherited from Superclass. */
		typedef typename Superclass::TransformType            TransformType;
		typedef typename Superclass::TransformPointer         TransformPointer;
		typedef typename Superclass::TransformJacobianType    TransformJacobianType;
		typedef typename Superclass::InterpolatorType         InterpolatorType;
		typedef typename Superclass::MeasureType              MeasureType;
		typedef typename Superclass::DerivativeType           DerivativeType;
		typedef typename Superclass::ParametersType           ParametersType;
		typedef typename Superclass::FixedImageType           FixedImageType;
		typedef typename Superclass::MovingImageType          MovingImageType;
		typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
		typedef typename Superclass::MovingImageConstPointer  MovingImageCosntPointer;
		typedef typename Superclass::CoordinateRepresentationType
			CoordinateRepresentationType;

		typedef typename Superclass::BSplineTransformType			        BSplineTransformType;
		typedef typename Superclass::BSplineCombinationTransformType	BSplineCombinationTransformType;
    typedef typename BSplineTransformType::SpacingType            GridSpacingType;
		
		/** Index and Point typedef support. */
		typedef typename FixedImageType::IndexType            FixedImageIndexType;
		typedef typename FixedImageIndexType::IndexValueType  FixedImageIndexValueType;
		typedef typename MovingImageType::IndexType           MovingImageIndexType;
		typedef typename TransformType::InputPointType        FixedImagePointType;
		typedef typename TransformType::OutputPointType       MovingImagePointType;
		
		/** The fixed image dimension. */
		itkStaticConstMacro( FixedImageDimension, unsigned int,
			FixedImageType::ImageDimension );

		/** The moving image dimension. */
		itkStaticConstMacro( MovingImageDimension, unsigned int,
			MovingImageType::ImageDimension );
		
    /** Typedefs for the two metrics we combine. */
    typedef MattesMutualInformationImageToImageMetric2<
			FixedImageType,MovingImageType >                      MattesMutualInformationMetricType;
    typedef typename MattesMutualInformationMetricType
      ::Pointer                                             MattesMutualInformationPointer;
		typedef RigidityPenaltyTermMetric<
			itkGetStaticConstMacro( FixedImageDimension ),
			CoordinateRepresentationType >											  RigidityPenaltyTermMetricType;
		typedef typename RigidityPenaltyTermMetricType::Pointer	RigidityPenaltyTermPointer;

    /** Typedefs for the rigidity penalty term. */
		typedef typename RigidityPenaltyTermMetricType
			::CoefficientImageType											        RigidityImageType;
		typedef typename RigidityPenaltyTermMetricType
			::CoefficientImagePointer														RigidityImagePointer;
		typedef typename RigidityPenaltyTermMetricType
			::ScalarType															          RigidityPixelType;
		typedef typename RigidityImageType::RegionType				RigidityImageRegionType;
		typedef typename RigidityImageType::IndexType					RigidityImageIndexType;
		typedef typename RigidityImageType::PointType					RigidityImagePointType;
		typedef ImageRegionIterator< RigidityImageType >			RigidityImageIteratorType;
		typedef BinaryBallStructuringElement<
			RigidityPixelType,
			itkGetStaticConstMacro( FixedImageDimension ) >			StructuringElementType;
		typedef typename StructuringElementType::RadiusType		SERadiusType;
		typedef GrayscaleDilateImageFilter<
			RigidityImageType, RigidityImageType,
			StructuringElementType >														DilateFilterType;
		typedef typename DilateFilterType::Pointer						DilateFilterPointer;
	
		/** Initialize the metric. */
		void Initialize(void) throw ( ExceptionObject );
		
		/**  Get the value. */
		MeasureType GetValue( const ParametersType& parameters ) const
    {
      this->FillRigidityCoefficientImage( parameters );
      return this->Superclass::GetValue( parameters );
    };

		/** Get the derivatives of the match measure. */
		void GetDerivative( const ParametersType& parameters,
			DerivativeType & derivative ) const
    {
      this->FillRigidityCoefficientImage( parameters );
      this->Superclass::GetDerivative( parameters, derivative );
    };

		/**  Get the value and derivatives for single valued optimizers. */
		void GetValueAndDerivative( const ParametersType& parameters, 
			MeasureType& value, DerivativeType& derivative ) const
    {
      this->FillRigidityCoefficientImage( parameters );
      this->Superclass::GetValueAndDerivative( parameters, value, derivative );
    };

		/** Set if the RigidityImage's are dilated. */
		itkSetMacro( DilateRigidityImages, bool );

		/** Set the DilationRadiusMultiplier. */
		itkSetClampMacro( DilationRadiusMultiplier, CoordinateRepresentationType,
			0.1, NumericTraits<CoordinateRepresentationType>::max() );

		/** Set the fixed coefficient image. */
		itkSetObjectMacro( FixedRigidityImage, RigidityImageType );

		/** Set the moving coefficient image. */
		itkSetObjectMacro( MovingRigidityImage, RigidityImageType );

		/** Set to use the FixedRigidityImage or not. */
		itkSetMacro( UseFixedRigidityImage, bool );

		/** Set to use the MovingRigidityImage or not. */
		itkSetMacro( UseMovingRigidityImage, bool );

		/** Function to fill the RigidityCoefficientImage every iteration. */
		void FillRigidityCoefficientImage( const ParametersType& parameters ) const;

		/** Set the OutputDirectoryName. *
		void SetOutputDirectoryName( const char * _arg );*/

	protected:
		
		/** The constructor. */
		MattesMutualInformationImageToImageMetricWithRigidityPenalty();
		/** The destructor. */
		virtual ~MattesMutualInformationImageToImageMetricWithRigidityPenalty() {};

		/** PrintSelf. */
		void PrintSelf( std::ostream& os, Indent indent ) const;

    /** The two metrics. */
    MattesMutualInformationPointer  m_MattesMutualInformationMetric;
		RigidityPenaltyTermPointer			m_RigidityPenaltyTermMetric;
		
	private:
		
		/** The private constructor. */
		MattesMutualInformationImageToImageMetricWithRigidityPenalty( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );																		// purposely not implemented
		
		/** Rigidity image variables. */
    CoordinateRepresentationType		m_DilationRadiusMultiplier;
		bool														m_DilateRigidityImages;
		RigidityImagePointer						m_FixedRigidityImage;
		RigidityImagePointer						m_MovingRigidityImage;
		RigidityImagePointer						m_RigidityCoefficientImage;
		std::vector< DilateFilterPointer >	m_FixedRigidityImageDilation;
		std::vector< DilateFilterPointer >	m_MovingRigidityImageDilation;
		RigidityImagePointer						m_FixedRigidityImageDilated;
		RigidityImagePointer						m_MovingRigidityImageDilated;
		bool														m_UseFixedRigidityImage;
		bool														m_UseMovingRigidityImage;

		/** Name of the output directory. *
		std::string m_OutputDirectoryName;*/
		
	}; // end class MattesMutualInformationImageToImageMetricWithRigidityPenalty

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMattesMutualInformationImageToImageMetricWithRigidityPenalty.hxx"
#endif

#endif // end #ifndef __itkMattesMutualInformationImageToImageMetricWithRigidityPenalty_H__

