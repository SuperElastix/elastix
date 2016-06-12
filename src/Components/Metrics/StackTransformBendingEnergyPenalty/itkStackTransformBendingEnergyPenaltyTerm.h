#ifndef __itkStackTransformBendingEnergyPenaltyTerm_h
#define __itkStackTransformBendingEnergyPenaltyTerm_h

#include "itkTransformPenaltyTerm.h"
#include "itkImageGridSampler.h"

#include "itkStackTransform.h"

namespace itk
{

template< class TFixedImage, class TScalarType >
class StackTransformBendingEnergyPenaltyTerm :
  public TransformPenaltyTerm< TFixedImage, TScalarType >
{
public:

  typedef StackTransformBendingEnergyPenaltyTerm Self;
  typedef TransformPenaltyTerm<
    TFixedImage, TScalarType >                  Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  itkNewMacro( Self );

  itkTypeMacro( StackTransformBendingEnergyPenaltyTerm, TransformPenaltyTerm );

  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType              MovingImageType;
  typedef typename Superclass::MovingImagePixelType         MovingImagePixelType;
  typedef typename Superclass::MovingImagePointer           MovingImagePointer;
  typedef typename Superclass::MovingImageConstPointer      MovingImageConstPointer;
  typedef typename Superclass::FixedImageType               FixedImageType;
  typedef typename Superclass::FixedImagePointer            FixedImagePointer;
  typedef typename Superclass::FixedImageConstPointer       FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType         FixedImageRegionType;
    typedef typename FixedImageType::SizeType                       FixedImageSizeType;
  typedef typename Superclass::TransformType                TransformType;
  typedef typename Superclass::TransformPointer             TransformPointer;
  typedef typename Superclass::InputPointType               InputPointType;
  typedef typename Superclass::OutputPointType              OutputPointType;
  typedef typename Superclass::TransformParametersType      TransformParametersType;
  typedef typename Superclass::TransformJacobianType        TransformJacobianType;
  typedef typename Superclass::NumberOfParametersType       NumberOfParametersType;
  typedef typename Superclass::InterpolatorType             InterpolatorType;
  typedef typename Superclass::InterpolatorPointer          InterpolatorPointer;
  typedef typename Superclass::RealType                     RealType;
  typedef typename Superclass::GradientPixelType            GradientPixelType;
  typedef typename Superclass::GradientImageType            GradientImageType;
  typedef typename Superclass::GradientImagePointer         GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType      GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer   GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType           FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer        FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType          MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer       MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                  MeasureType;
  typedef typename Superclass::DerivativeType               DerivativeType;
  typedef typename Superclass::DerivativeValueType          DerivativeValueType;
  typedef typename Superclass::ParametersType               ParametersType;
  typedef typename Superclass::FixedImagePixelType          FixedImagePixelType;
  typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer  ImageSampleContainerPointer;
  typedef typename Superclass::ScalarType                   ScalarType;
  typedef typename Superclass::ThreaderType                 ThreaderType;
  typedef typename Superclass::ThreadInfoType               ThreadInfoType;


    itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
    
    itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );
    
    itkStaticConstMacro( ReducedFixedImageDimension, unsigned int, FixedImageType::ImageDimension - 1 );
    
    itkStaticConstMacro( ReducedMovingImageDimension, unsigned int, MovingImageType::ImageDimension - 1 );

  typedef itk::StackTransform< ScalarType, FixedImageDimension, MovingImageDimension >        StackTransformType;

  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;

  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;
  typedef typename Superclass::HessianValueType   HessianValueType;
  typedef typename Superclass::HessianType        HessianType;


  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  virtual void GetDerivative( const ParametersType & parameters,
    DerivativeType & derivative ) const;

  virtual void GetValueAndDerivativeSingleThreaded(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

  virtual void GetValueAndDerivative(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

  inline void ThreadedGetValueAndDerivative( ThreadIdType threadID );

  inline void AfterThreadedGetValueAndDerivative(
    MeasureType & value, DerivativeType & derivative ) const;
    
    itkSetMacro( SubtractMean, bool );
    itkSetMacro( TransformIsStackTransform, bool );
    itkSetMacro( TransformIsBSpline, bool );
    itkSetMacro( SubTransformIsBSpline, bool );
    itkSetMacro( GridSize, FixedImageSizeType );

    itkSetMacro( SampleLastDimensionRandomly, bool );
    itkSetMacro( NumSamplesLastDimension, unsigned int );

protected:

  typedef typename Superclass::FixedImageIndexType            FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType       FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType           MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType            FixedImagePointType;
  typedef typename Superclass::MovingImagePointType           MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType MovingImageContinuousIndexType;
  typedef typename Superclass::NonZeroJacobianIndicesType     NonZeroJacobianIndicesType;
  typedef typename itk::ContinuousIndex< CoordinateRepresentationType, FixedImageDimension >  FixedImageContinuousIndexType;
    
  typedef typename Superclass::BSplineTransformType     BSplineTransformType;
  typedef typename Superclass::CombinationTransformType CombinationTransformType;

  StackTransformBendingEnergyPenaltyTerm();

  virtual ~StackTransformBendingEnergyPenaltyTerm() {}
    
    void SampleRandom( const int n, const int m, std::vector< int > & numbers ) const;
    
    bool m_TransformIsStackTransform;
    bool m_SubTransformIsBSpline;
    bool m_TransformIsBSpline;
    bool m_SubtractMean;

    FixedImageSizeType m_GridSize;

private:

  StackTransformBendingEnergyPenaltyTerm( const Self & ); // purposely not implemented
  void operator=( const Self & );                    // purposely not implemented

    mutable unsigned int m_NumSamplesLastDimension;
    
    bool m_SampleLastDimensionRandomly;
    mutable std::vector<int> m_RandomList;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkStackTransformBendingEnergyPenaltyTerm.hxx"
#endif

#endif // #ifndef __itkStackTransformBendingEnergyPenaltyTerm_h
