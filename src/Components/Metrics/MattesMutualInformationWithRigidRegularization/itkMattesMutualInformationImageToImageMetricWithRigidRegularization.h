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
#ifndef __itkMattesMutualInformationImageToImageMetricWithRigidRegularization_H__
#define __itkMattesMutualInformationImageToImageMetricWithRigidRegularization_H__

#include "../MattesMutualInformation/itkMattesMutualInformationImageToImageMetricWithMask.h"

/** Include the penalty term. */
#include "itkRigidRegulizerMetric.h"

#include "itkGrayscaleDilateImageFilter.h"
#include "itkBinaryBallStructuringElement.h"

#include "itkImageRegionIterator.h"

namespace itk
{
	
	/** \class MattesMutualInformationImageToImageMetricWithRigidRegularization
	* \brief Computes the mutual information between two images to be 
	* registered using the method of Mattes et al. and adds a rigid penalty term.
	*
	* MattesMutualInformationImageToImageMetricWithRigidRegularization computes the mutual 
	* information between a fixed and moving image to be registered and adds a
	* rigid penalty term. The rigid penalty term penalizes deviations from a rigid
	* transformation at regions specified by the so-called rigidity images.
	*
	* This class is derived from MattesMutualInformationImageToImageMetricWithMask,
	* which is the itk::MattesMutualInformationImageToImageMetric with some
	* contributions of our own. This class changes the GetValue() and 
	* GetValueAndDerivative() methods, such that it adds a rigid penalty
	* term to the Mutual Information metric of the superclass.
	*
	* This metric only works with B-splines as a transformation model.
	*
	* \sa MattesMutualInformationImageToImageMetricWithMask
	* \sa MattesMutualInformationMetricWithRigidRegularization
	* \sa RigidRegulizerMetric
	* \sa BSplineTransform
	* \ingroup Metrics
	*/

	template < class TFixedImage, class TMovingImage >
		class MattesMutualInformationImageToImageMetricWithRigidRegularization :
	public MattesMutualInformationImageToImageMetricWithMask< TFixedImage, TMovingImage >
	{

	public:
		
		/** Standard class typedefs. */
		typedef MattesMutualInformationImageToImageMetricWithRigidRegularization		Self;
		typedef MattesMutualInformationImageToImageMetricWithMask<
			TFixedImage, TMovingImage >													Superclass;
		typedef SmartPointer<Self>														Pointer;
		typedef SmartPointer<const Self>											ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( MattesMutualInformationImageToImageMetricWithRigidRegularization,
			MattesMutualInformationImageToImageMetricWithMask );
		
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
		
		/** Typedef's for the rigid penalty term. */
		typedef RigidRegulizerMetric<
			itkGetStaticConstMacro( FixedImageDimension ),
			CoordinateRepresentationType >											RigidRegulizerMetricType;
		typedef typename RigidRegulizerMetricType::Pointer		RigidRegulizerMetricPointer;
		typedef typename RigidRegulizerMetricType
			::RigidityImageType																	RigidityImageType;
		typedef typename RigidRegulizerMetricType
			::RigidityImagePointer															RigidityImagePointer;
		typedef typename RigidRegulizerMetricType
			::CoefficientPixelType															RigidityPixelType;
		typedef typename RigidityImageType::RegionType				RigidityImageRegionType;
		typedef typename RigidityImageType::IndexType					RigidityImageIndexType;
		typedef typename RigidityImageType::PointType					RigidityImagePointType;
		typedef ImageRegionIterator< RigidityImageType >			RigidityImageIteratorType;
		typedef BinaryBallStructuringElement<
			RigidityPixelType,
			itkGetStaticConstMacro( FixedImageDimension ) >			StructuringElementType;
		typedef typename StructuringElementType::RadiusType		SERadiusType;
		//typedef typename RadiusType::SizeValueType					RadiusValueType;
		typedef GrayscaleDilateImageFilter<
			RigidityImageType, RigidityImageType,
			StructuringElementType >														DilateFilterType;
		typedef typename DilateFilterType::Pointer						DilateFilterPointer;

		/** Initialize the metric. */
		void Initialize(void) throw ( ExceptionObject );
		
		/**  Get the value. */
		MeasureType GetValue( const ParametersType& parameters ) const;

		/** Get the derivatives of the match measure. */
		void GetDerivative( const ParametersType& parameters,
			DerivativeType & derivative ) const;

		/**  Get the value and derivatives for single valued optimizers. */
		void GetValueAndDerivative( const ParametersType& parameters, 
			MeasureType& value, DerivativeType& derivative ) const;

		/** Set the weighting between the mutual information and
		 * the rigid penalty term.
		 */
		itkSetClampMacro( RigidPenaltyWeight, CoordinateRepresentationType,
			0.0, NumericTraits<CoordinateRepresentationType>::max() );
		itkGetMacro( RigidPenaltyWeight, CoordinateRepresentationType );

		/** Set the weighting between the first and second order
		 * information of the rigid penalty term.
		 */
		itkSetClampMacro( SecondOrderWeight, CoordinateRepresentationType,
			0.0, NumericTraits<CoordinateRepresentationType>::max() );

		/** Set if the image spacing is used. */
		itkSetMacro( UseImageSpacing, bool );

		/** Set if the RigidityImage's are dilated. */
		itkSetMacro( DilateRigidityImages, bool );

		/** Set the DilationRadiusMultiplier. */
		itkSetClampMacro( DilationRadiusMultiplier, CoordinateRepresentationType,
			0.1, NumericTraits<CoordinateRepresentationType>::max() );

		/** Set the fixed coefficient image. */
		itkSetMacro( FixedRigidityImage, RigidityImagePointer );

		/** Set the moving coefficient image. */
		itkSetMacro( MovingRigidityImage, RigidityImagePointer );

		/** Set to use the FixedRigidityImage or not. */
		itkSetMacro( UseFixedRigidityImage, bool );

		/** Set to use the MovingRigidityImage or not. */
		itkSetMacro( UseMovingRigidityImage, bool );

		/** Function to fill the RigidityCoefficientImage every iteration. */
		void FillRigidityCoefficientImage( const ParametersType& parameters ) const;

		/** For printing purposes. */
		itkGetMacro( MIValue, double );
		itkGetMacro( RigidValue, double );

		/** Set the OutputDirectoryName. */
		void SetOutputDirectoryName( const char * _arg );

	protected:
		
		/** The constructor. */
		MattesMutualInformationImageToImageMetricWithRigidRegularization();
		/** The destructor. */
		virtual ~MattesMutualInformationImageToImageMetricWithRigidRegularization() {};

		/** PrintSelf. */
		void PrintSelf( std::ostream& os, Indent indent ) const;
		
	private:
		
		/** The private constructor. */
		MattesMutualInformationImageToImageMetricWithRigidRegularization( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );																		// purposely not implemented
		
		/** Weightings. */
		CoordinateRepresentationType		m_RigidPenaltyWeight;
		CoordinateRepresentationType		m_SecondOrderWeight;
		CoordinateRepresentationType		m_DilationRadiusMultiplier;
		bool														m_UseImageSpacing;
		bool														m_DilateRigidityImages;

		/** The rigid regulizer metric. */
		RigidRegulizerMetricPointer			m_RigidRegulizer;

		/** Rigidity image variables. */
		RigidityImagePointer						m_FixedRigidityImage;
		RigidityImagePointer						m_MovingRigidityImage;
		RigidityImagePointer						m_RigidityCoefficientImage;
		std::vector< DilateFilterPointer >	m_FixedRigidityImageDilation;
		std::vector< DilateFilterPointer >	m_MovingRigidityImageDilation;
		RigidityImagePointer						m_FixedRigidityImageDilated;
		RigidityImagePointer						m_MovingRigidityImageDilated;
		bool														m_UseFixedRigidityImage;
		bool														m_UseMovingRigidityImage;

		/** For printing purposes. */
		mutable double m_MIValue, m_RigidValue;

		/** Name of the output directory. */
		std::string m_OutputDirectoryName;
		
	}; // end class MattesMutualInformationImageToImageMetricWithRigidRegularization

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMattesMutualInformationImageToImageMetricWithRigidRegularization.hxx"
#endif

#endif // end #ifndef __itkMattesMutualInformationImageToImageMetricWithRigidRegularization_H__

