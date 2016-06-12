#ifndef __elxLinearGroupwiseMSD_H__
#define __elxLinearGroupwiseMSD_H__

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkLinearGroupwiseMSD.h"

#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkStackTransform.h"

namespace elastix
{
    
    template< class TElastix >
    class LinearGroupwiseMSD : public itk::LinearGroupwiseMSD< typename MetricBase< TElastix >::FixedImageType, typename MetricBase< TElastix >::MovingImageType >, public MetricBase< TElastix >
    {
        public:
        
            typedef LinearGroupwiseMSD      Self;
            typedef itk::LinearGroupwiseMSD< typename MetricBase< TElastix >::FixedImageType,typename MetricBase< TElastix >::MovingImageType >   Superclass1;
            typedef MetricBase< TElastix >    Superclass2;
            typedef itk::SmartPointer< Self >     Pointer;
            typedef itk::SmartPointer< const Self >    ConstPointer;
        
            itkNewMacro( Self );

            itkTypeMacro( LinearGroupwiseMSD, itk::LinearGroupwiseMSD );
        
            elxClassNameMacro( "LinearGroupwiseMSD" );
        
        
            typedef typename Superclass1::CoordinateRepresentationType      CoordinateRepresentationType;
            typedef typename Superclass1::ScalarType                        ScalarType;
            typedef typename Superclass1::MovingImageType                   MovingImageType;
            typedef typename Superclass1::MovingImagePixelType              MovingImagePixelType;
            typedef typename Superclass1::MovingImageConstPointer           MovingImageConstPointer;
            typedef typename Superclass1::FixedImageType                    FixedImageType;
            typedef typename FixedImageType::SizeType                       FixedImageSizeType;
            typedef typename Superclass1::FixedImageConstPointer            FixedImageConstPointer;
            typedef typename Superclass1::FixedImageRegionType              FixedImageRegionType;
            typedef typename Superclass1::TransformType                     TransformType;
            typedef typename Superclass1::TransformPointer                  TransformPointer;
            typedef typename Superclass1::InputPointType                    InputPointType;
            typedef typename Superclass1::OutputPointType                   OutputPointType;
            typedef typename Superclass1::TransformParametersType           TransformParametersType;
            typedef typename Superclass1::TransformJacobianType             TransformJacobianType;
            typedef typename Superclass1::InterpolatorType                  InterpolatorType;
            typedef typename Superclass1::InterpolatorPointer               InterpolatorPointer;
            typedef typename Superclass1::RealType                          RealType;
            typedef typename Superclass1::GradientPixelType                 GradientPixelType;
            typedef typename Superclass1::GradientImageType                 GradientImageType;
            typedef typename Superclass1::GradientImagePointer              GradientImagePointer;
            typedef typename Superclass1::GradientImageFilterType           GradientImageFilterType;
            typedef typename Superclass1::GradientImageFilterPointer        GradientImageFilterPointer;
            typedef typename Superclass1::FixedImageMaskType                FixedImageMaskType;
            typedef typename Superclass1::FixedImageMaskPointer             FixedImageMaskPointer;
            typedef typename Superclass1::MovingImageMaskType               MovingImageMaskType;
            typedef typename Superclass1::MovingImageMaskPointer            MovingImageMaskPointer;
            typedef typename Superclass1::MeasureType                       MeasureType;
            typedef typename Superclass1::DerivativeType                    DerivativeType;
            typedef typename Superclass1::ParametersType                    ParametersType;
            typedef typename Superclass1::FixedImagePixelType               FixedImagePixelType;
            typedef typename Superclass1::MovingImageRegionType             MovingImageRegionType;
            typedef typename Superclass1::ImageSamplerType                  ImageSamplerType;
            typedef typename Superclass1::ImageSamplerPointer               ImageSamplerPointer;
            typedef typename Superclass1::ImageSampleContainerType          ImageSampleContainerType;
            typedef typename Superclass1::ImageSampleContainerPointer       ImageSampleContainerPointer;
            typedef typename Superclass1::FixedImageLimiterType             FixedImageLimiterType;
            typedef typename Superclass1::MovingImageLimiterType            MovingImageLimiterType;
            typedef typename Superclass1::FixedImageLimiterOutputType       FixedImageLimiterOutputType;
            typedef typename Superclass1::MovingImageLimiterOutputType      MovingImageLimiterOutputType;
            typedef typename Superclass1::MovingImageDerivativeScalesType   MovingImageDerivativeScalesType;
        
        itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
        
        itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );

        
        typedef itk::AdvancedBSplineDeformableTransformBase<
        ScalarType, FixedImageDimension >                       BSplineTransformBaseType;
        typedef itk::AdvancedCombinationTransform<
        ScalarType, FixedImageDimension >                       CombinationTransformType;
        typedef itk::StackTransform<
        ScalarType, FixedImageDimension, MovingImageDimension > StackTransformType;
        typedef itk::AdvancedBSplineDeformableTransformBase<
        ScalarType, FixedImageDimension - 1 >                   ReducedDimensionBSplineTransformBaseType;

            typedef typename Superclass2::ElastixType          ElastixType;
            typedef typename Superclass2::ElastixPointer       ElastixPointer;
            typedef typename Superclass2::ConfigurationType    ConfigurationType;
            typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
            typedef typename Superclass2::RegistrationType     RegistrationType;
            typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
            typedef typename Superclass2::ITKBaseType          ITKBaseType;
                        
            virtual void Initialize( void ) throw ( itk::ExceptionObject );
            virtual void BeforeEachResolution( void );
            virtual void BeforeRegistration( void );
        
        protected:
        
            LinearGroupwiseMSD();
        
            virtual ~LinearGroupwiseMSD() {}
        
        private:
        
            LinearGroupwiseMSD( const Self & );  // purposely not implemented
            void operator=( const Self & );                 // purposely not implemented
        
    };
    
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxLinearGroupwiseMSD.hxx"
#endif

#endif // end #ifndef __elxLinearGroupwiseMSD_H__
