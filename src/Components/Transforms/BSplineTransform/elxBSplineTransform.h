#ifndef __elxBSplineTransform_h
#define __elxBSplineTransform_h

/* For easy changing the BSplineOrder: */
#define __VSplineOrder 3
/* For easy changing the PixelType of the saved deformation fields: */
#define __CoefficientOutputType float

#include "itkBSplineDeformableTransform.h"
#include "itkBSplineResampleImageFilterBase.h"
#include "itkBSplineUpsampleImageFilter.h"
#include "itkImage.h"
#include "itkCastImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "elxIncludes.h"

#include "itkBSplineTransformGrouper.h"

#include <sstream>

namespace elastix
{
using namespace itk;


	/**
	 * ********************* BSplineTransform ***********************
	 *
	 * This class
	 */

	template < class TElastix >
		class BSplineTransform:
	public
		BSplineTransformGrouper< 
			BSplineDeformableTransform<
				ITK_TYPENAME TransformBase<TElastix>::CoordRepType,
				TransformBase<TElastix>::FixedImageDimension,
				__VSplineOrder >		>,
	public
		TransformBase<TElastix>
	{
	public:

		/** Standard ITK-stuff.*/
		typedef BSplineTransform 										Self;
		typedef BSplineDeformableTransform<
			typename TransformBase<TElastix>::CoordRepType,
			TransformBase<TElastix>::FixedImageDimension,
			__VSplineOrder >													Superclass1;
		typedef TransformBase<TElastix>							Superclass2;		
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( BSplineTransform, BSplineDeformableTransform );

		/** Name of this class.*/
		elxClassNameMacro( "BSplineTransform" );
		
		/** Dimension of the domain space. */
		itkStaticConstMacro( SpaceDimension, unsigned int, Superclass2::FixedImageDimension );
		
		/** The BSpline order. */
		itkStaticConstMacro( SplineOrder, unsigned int, __VSplineOrder );
		
		/** Typedefs inherited from the superclass.*/
		typedef typename Superclass1::ScalarType 								ScalarType;
		typedef typename Superclass1::ParametersType 						ParametersType;
		typedef typename Superclass1::JacobianType 							JacobianType;
		typedef typename Superclass1::InputVectorType						InputVectorType;
		typedef typename Superclass1::OutputVectorType 					OutputVectorType;
		typedef typename Superclass1::InputCovariantVectorType 	InputCovariantVectorType;
		typedef typename Superclass1::OutputCovariantVectorType	OutputCovariantVectorType;
		typedef typename Superclass1::InputVnlVectorType 				InputVnlVectorType;
		typedef typename Superclass1::OutputVnlVectorType				OutputVnlVectorType;
		typedef typename Superclass1::InputPointType 						InputPointType;
		typedef typename Superclass1::OutputPointType						OutputPointType;
		
		/** Typedef's specific for the BSplineTransform.*/
		typedef typename Superclass1::PixelType									PixelType;
		typedef typename Superclass1::ImageType									ImageType;
		typedef typename Superclass1::ImagePointer							ImagePointer;
		typedef typename Superclass1::RegionType								RegionType;
		typedef typename Superclass1::IndexType									IndexType;
		typedef typename Superclass1::SizeType									SizeType;
		typedef typename Superclass1::SpacingType								SpacingType;
		typedef typename Superclass1::OriginType								OriginType;
		typedef typename Superclass1::BulkTransformType					BulkTransformType;
		typedef typename Superclass1::BulkTransformPointer			BulkTransformPointer;
		typedef typename Superclass1::WeightsFunctionType				WeightsFunctionType;
		typedef typename Superclass1::WeightsType								WeightsType;
		typedef typename Superclass1::ContinuousIndexType				ContinuousIndexType;
		typedef typename Superclass1::ParameterIndexArrayType		ParameterIndexArrayType;

		/** Typedef's from TransformBase.*/
		typedef typename Superclass2::ElastixType								ElastixType;
		typedef typename Superclass2::ElastixPointer						ElastixPointer;
		typedef typename Superclass2::ConfigurationType					ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer			ConfigurationPointer;
		typedef typename Superclass2::RegistrationType					RegistrationType;
		typedef typename Superclass2::RegistrationPointer				RegistrationPointer;
		typedef typename Superclass2::CoordRepType							CoordRepType;
		typedef typename Superclass2::FixedImageType						FixedImageType;
		typedef typename Superclass2::MovingImageType						MovingImageType;
		typedef typename Superclass2::ITKBaseType								ITKBaseType;

		/** Typedefs & Vars needed for setting the grid size and upsampling the grid.*/
		
		/** the FixedImagePyramidBase */
		typedef typename ElastixType::FixedImagePyramidBaseType			FixedImagePyramidType;

		/** itk::MultiResolutionImagePyramidFilter */
		typedef typename FixedImagePyramidType::ITKBaseType			FixedImagePyramidITKBaseType;

		/** pointer to itk::MultiResolutionImagePyramidFilter  */
		typedef FixedImagePyramidITKBaseType *									FixedImagePyramidPointer;
		
		/** Other typedef's.*/
		typedef BSplineResampleImageFilterBase<
			ImageType, ImageType>												BSplineResamplerType;
		typedef BSplineUpsampleImageFilter<
			ImageType,ImageType,BSplineResamplerType>		UpsamplerType; 
		typedef ImageRegionConstIterator<ImageType>		IteratorType;

		/** For saving the deformation fields: */
		typedef __CoefficientOutputType								CoefficientOutputType;
		typedef Image< CoefficientOutputType,
			itkGetStaticConstMacro(SpaceDimension) >		CoefficientOutputImageType;
		typedef CastImageFilter< 
			ImageType, CoefficientOutputImageType >			TransformCastFilterType;
		typedef ImageFileWriter<
			CoefficientOutputImageType >								TransformWriterType;

		/** Methods that have to be present in each version of BSplineTransform.*/
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);
		
		/** These two methods are called by BeforeEachResolution()*/
		virtual void SetInitialGrid( bool upsampleGridOption );
		virtual void IncreaseScale(void);
		
		/** Function to read/write transform-parameters from/to a file.*/
		virtual void ReadFromFile(void);
		virtual void WriteToFile( const ParametersType & param );
		
	protected:

		BSplineTransform();
		virtual ~BSplineTransform() {std::cerr << "transform dies" << std::endl;};
		
		/** Member variabels.*/
		typename UpsamplerType::Pointer							m_Upsampler;
		typename TransformCastFilterType::Pointer		m_Caster;
		typename TransformWriterType::Pointer				m_Writer;
				
		ParametersType * m_Parameterspointer;
	  ParametersType * m_Parameterspointer_out;
	
		ImagePointer m_Coeffs1;
	  ImagePointer m_Coeffs2;

	private:
		BSplineTransform( const Self& );	// purposely not implemented
		void operator=( const Self& );		// purposely not implemented
		
	}; // end class BSplineTransform
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxBSplineTransform.hxx"
#endif

#endif // end #ifndef __elxBSplineTransform_h

