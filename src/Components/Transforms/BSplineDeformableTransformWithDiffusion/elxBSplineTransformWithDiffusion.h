#ifndef __elxBSplineTransformWithDiffusion_H__
#define __elxBSplineTransformWithDiffusion_H__

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

/** Include structure for the diffusion. */
#include "itkDeformationFieldRegulizerForBSpline.h"
#include "itkVectorMeanDiffusionImageFilter.h"
#include "itkResampleImageFilter.h"

namespace elastix
{
using namespace itk;


	/**
	 * \class BSplineTransformWithDiffusion
	 * \brief This class combines a B-spline transform with the
	 * diffusion of the deformationfield.
	 *
	 * Every n iterations the deformationfield is diffused using the
	 * VectorMeanDiffusionImageFilter. The total transformation of a point
	 * is determined by adding the bspline deformation to the
	 * deformationfield arrow.
	 *
	 * \ingroup Transforms
	 */

	template < class TElastix >
		class BSplineTransformWithDiffusion:
	public
	DeformationFieldRegulizerForBSpline<
		BSplineTransformGrouper<
			BSplineDeformableTransform<
				ITK_TYPENAME TransformBase<TElastix>::CoordRepType,
				TransformBase<TElastix>::FixedImageDimension,
				__VSplineOrder
			>
		>
	>,
	public
		TransformBase<TElastix>
	{
	public:

		/** Standard ITK-stuff.*/
		typedef BSplineTransformWithDiffusion				Self;
		typedef DeformationFieldRegulizerForBSpline<
			BSplineTransformGrouper<
				BSplineDeformableTransform<
					typename TransformBase<TElastix>::CoordRepType,
					TransformBase<TElastix>::FixedImageDimension,
					__VSplineOrder > > >									Superclass1;
		typedef TransformBase<TElastix>							Superclass2;
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( BSplineTransformWithDiffusion, DeformationFieldRegulizerForBSpline );

		/** Name of this class.*/
		elxClassNameMacro( "BSplineTransformWithDiffusion" );
		
		/** Dimension of the domain space. */
		itkStaticConstMacro( SpaceDimension, unsigned int, Superclass2::FixedImageDimension );
		
		/** The BSpline order. */
		itkStaticConstMacro( SplineOrder, unsigned int, __VSplineOrder );
		
		/** Typedefs inherited from the superclass. */
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
		
		/** Typedef's specific for the BSplineTransform. */
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

		/** Typedef's from TransformBase. */
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
		typedef typename ElastixType::FixedImagePyramidBaseType	FixedImagePyramidType;
		typedef typename ElastixType::FixedImageType						FixedImageELXType;
		typedef typename ElastixType::MovingImageType						MovingImageELXType;

		/** itk::MultiResolutionImagePyramidFilter */
		typedef typename FixedImagePyramidType::ITKBaseType			FixedImagePyramidITKBaseType;

		/** pointer to itk::MultiResolutionImagePyramidFilter  */
		typedef FixedImagePyramidITKBaseType *									FixedImagePyramidPointer;
		
		/** Other typedef's.*/
		typedef	Image< short,
			itkGetStaticConstMacro( SpaceDimension ) >	DummyImageType;
		typedef ImageRegionConstIterator<
			DummyImageType >														DummyIteratorType;

		/** For saving the deformation fields: */
		typedef __CoefficientOutputType								CoefficientOutputType;
		typedef Image< CoefficientOutputType,
			itkGetStaticConstMacro(SpaceDimension) >		CoefficientOutputImageType;
		typedef CastImageFilter< 
			ImageType, CoefficientOutputImageType >			TransformCastFilterType;
		typedef ImageFileWriter<
			CoefficientOutputImageType >								TransformWriterType;

		/** Typedef's for the diffusion of the deformation field. */
		typedef typename Superclass2::OutputImageType			VectorImageType;
		typedef typename VectorImageType::PixelType				VectorType;
		typedef ImageRegionIterator<
			VectorImageType >																VectorImageIteratorType;
		typedef FixedImageELXType													GrayValueImageType;
		typedef VectorMeanDiffusionImageFilter<
			VectorImageType, GrayValueImageType >						DiffusionFilterType;
		typedef typename VectorImageType::SizeType				RadiusType;
		typedef ResampleImageFilter<
			MovingImageELXType, GrayValueImageType,
			CoordRepType >																	ResamplerType;
		typedef ImageFileWriter< GrayValueImageType >			GrayValueImageWriterType;
		typedef ImageFileWriter< VectorImageType >				DeformationFieldWriterType;

		/** Execute stuff before the actual registration:
		 * \li Create an initial B-spline grid.
		 * \li Create initial registration parameters.
		 * \li Setup stuff for the diffusion of the deformation field.
		 */
		virtual void BeforeRegistration(void);

		/** Execute stuff before each new pyramid resolution:
		 * \li upsample the B-spline grid.
		 */
		virtual void BeforeEachResolution(void);

		/** Execute stuff after each iteration:
		 * \li Do a diffusion of the deformation field.
		 */
		virtual void AfterEachIteration(void);
		
		/** Set the initial B-spline grid. */
		virtual void SetInitialGrid( bool upsampleGridOption );

		/** Upsample the B-spline grid. */
		virtual void IncreaseScale(void);
		
		/** Function to read transform-parameters from a file. */
		virtual void ReadFromFile(void);
		/** Function to write transform-parameters to a file. */
		virtual void WriteToFile( const ParametersType & param );
		
		/** Diffuse the deformation field. */
		void DiffuseDeformationField(void);

	protected:

		/** The constructor. */
		BSplineTransformWithDiffusion();
		/** The destructor. */
		virtual ~BSplineTransformWithDiffusion() {};
		
		/** Member variables.*/
		typename TransformCastFilterType::Pointer		m_Caster;
		typename TransformWriterType::Pointer				m_Writer;
		double																			m_GridSpacingFactor;

		ParametersType * m_Parameterspointer;
	  ParametersType * m_Parameterspointer_out;
	
		ImagePointer m_Coeffs1;
	  ImagePointer m_Coeffs2;

	private:

		/** The private constructor. */
		BSplineTransformWithDiffusion( const Self& );	// purposely not implemented
		void operator=( const Self& );								// purposely not implemented
		
		/** Member variables for diffusion. */
		typename DiffusionFilterType::Pointer		m_Diffusion;
		typename VectorImageType::Pointer				m_DeformationField;
		typename VectorImageType::Pointer				m_DiffusedField;
		typename GrayValueImageType::Pointer		m_GrayValueImage;
		typename ResamplerType::Pointer					m_Resampler;
		RegionType															m_DeformationRegion;
		OriginType															m_DeformationOrigin;
		SpacingType															m_DeformationSpacing;

		/** Member variables for writing diffusion files. */
		bool m_WriteDiffusionFiles;

	}; // end class BSplineTransformWithDiffusion
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxBSplineTransformWithDiffusion.hxx"
#endif

#endif // end #ifndef __elxBSplineTransformWithDiffusion_H__


