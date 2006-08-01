#ifndef __itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric_h
#define __itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric_h

#include "itkImageToImageMetricWithFeatures.h"

/** Includes for the kNN trees. */
#include "itkArray.h"
#include "itkListSampleCArray.h"
#include "itkBinaryTreeBase.h"
#include "itkBinaryTreeSearchBase.h"

/** Supported trees. */
#include "itkANNkDTree.h"
#include "itkANNbdTree.h"
#include "itkANNBruteForceTree.h"

/** Supported tree searchers. */
#include "itkANNStandardTreeSearch.h"
#include "itkANNFixedRadiusTreeSearch.h"
#include "itkANNPriorityTreeSearch.h"


namespace itk
{
/**
 * \class KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric
 *
 * \brief Computes similarity between two images to be registered.
 *
 * alpha mutual information, calculation based on binary trees. See
 * Neemuchwala.
 *
 * Note that the feature image are given beforehand, and that values
 * are calculated by interpolation on the transformed point. For some
 * features, it would be better (but slower) to first apply the transform
 * on the image and then recalculate the feature.
 * 
 * \ingroup RegistrationMetrics
 */
  
template < class TFixedImage, class TMovingImage,
  class TFixedFeatureImage = TFixedImage, class TMovingFeatureImage = TMovingImage>
class ITK_EXPORT KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric :
  public ImageToImageMetricWithFeatures< TFixedImage, TMovingImage, TFixedFeatureImage, TMovingFeatureImage>
{
public:

  /** Standard itk. */
  typedef KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric  Self;
  typedef ImageToImageMetricWithFeatures<
    TFixedImage, TMovingImage,
    TFixedFeatureImage, TMovingFeatureImage >                         Superclass;
  typedef SmartPointer<Self>                                          Pointer;
  typedef SmartPointer<const Self>                                    ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );
 
  /** Run-time type information (and related methods). */
  itkTypeMacro( KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric,
    ImageToImageMetricWithFeatures );
 
  /** Types transferred from the Superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType              MovingImageType;
  typedef typename Superclass::MovingImagePixelType         MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer      MovingImageConstPointer;
  typedef typename Superclass::FixedImageType               FixedImageType;
  typedef typename Superclass::FixedImageConstPointer       FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType         FixedImageRegionType;

  typedef typename Superclass::TransformType                TransformType;
  typedef typename Superclass::TransformPointer             TransformPointer;
  typedef typename Superclass::InputPointType               InputPointType;
  typedef typename Superclass::OutputPointType              OutputPointType;
  typedef typename Superclass::TransformParametersType      TransformParametersType;
  typedef typename Superclass::TransformJacobianType        TransformJacobianType;
  
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
  typedef typename Superclass::ParametersType               ParametersType;

  typedef typename Superclass::ImageSamplerType             ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer          ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer  ImageSampleContainerPointer;

  typedef typename Superclass::FixedFeatureImageType        FixedFeatureImageType;
  typedef typename Superclass::FixedFeatureImagePointer     FixedFeatureImagePointer;
  typedef typename Superclass::MovingFeatureImageType       MovingFeatureImageType;
  typedef typename Superclass::MovingFeatureImagePointer    MovingFeatureImagePointer;
  typedef typename Superclass::FixedFeatureImageVectorType  FixedFeatureImageVectorType;
  typedef typename Superclass::MovingFeatureImageVectorType MovingFeatureImageVectorType;

  typedef typename Superclass::FixedFeatureInterpolatorType         FixedFeatureInterpolatorType;
  typedef typename Superclass::MovingFeatureInterpolatorType        MovingFeatureInterpolatorType;
  typedef typename Superclass::FixedFeatureInterpolatorPointer      FixedFeatureInterpolatorPointer;
  typedef typename Superclass::MovingFeatureInterpolatorPointer     MovingFeatureInterpolatorPointer;
  typedef typename Superclass::FixedFeatureInterpolatorVectorType   FixedFeatureInterpolatorVectorType;
  typedef typename Superclass::MovingFeatureInterpolatorVectorType  MovingFeatureInterpolatorVectorType;

  /** The fixed image dimension. */
	itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );

  /** Typedefs for the samples. */
  typedef Array< double >                             MeasurementVectorType;
  typedef typename MeasurementVectorType::ValueType   MeasurementVectorValueType;
  typedef typename Statistics::ListSampleCArray<
    MeasurementVectorType, double >                   ListSampleType;
  
  /** Typedefs for trees. */
  typedef BinaryTreeBase< ListSampleType >            BinaryKNNTreeType;
  typedef ANNkDTree< ListSampleType >                 ANNkDTreeType;
  typedef ANNbdTree< ListSampleType >                 ANNbdTreeType;
  typedef ANNBruteForceTree< ListSampleType >         ANNBruteForceTreeType;

  /** Typedefs for tree searchers. */
  typedef BinaryTreeSearchBase< ListSampleType >      BinaryKNNTreeSearchType;
  typedef ANNStandardTreeSearch< ListSampleType >     ANNStandardTreeSearchType;
  typedef ANNFixedRadiusTreeSearch< ListSampleType >  ANNFixedRadiusTreeSearchType;
  typedef ANNPriorityTreeSearch< ListSampleType >     ANNPriorityTreeSearchType;

  typedef typename BinaryKNNTreeSearchType::IndexArrayType      IndexArrayType;
  typedef typename BinaryKNNTreeSearchType::DistanceArrayType   DistanceArrayType;

  /**
   * *** Set trees: ***
   * Currently kd, bd, and brute force trees are supported.
   */

  /** Set ANNkDTree. */
  void SetANNkDTree( unsigned int bucketSize, std::string splittingRule );

  /** Set ANNkDTree. */
  void SetANNkDTree( unsigned int bucketSize, std::string splittingRuleFixed,
    std::string splittingRuleMoving, std::string splittingRuleJoint );

  /** Set ANNbdTree. */
  void SetANNbdTree( unsigned int bucketSize, std::string splittingRule, std::string shrinkingRule );

  /** Set ANNbdTree. */
  void SetANNbdTree( unsigned int bucketSize, std::string splittingRuleFixed,
    std::string splittingRuleMoving, std::string splittingRuleJoint,
    std::string shrinkingRuleFixed, std::string shrinkingRuleMoving,
    std::string shrinkingRuleJoint );

  /** Set ANNBruteForceTree. */
  void SetANNBruteForceTree( void );

  /**
   * *** Set tree searchers: ***
   * Currently standard, fixed radius, and priority tree searchers are supported.
   */

  /** Set ANNStandardTreeSearch. */
  void SetANNStandardTreeSearch( unsigned int kNearestNeighbors, double errorBound );

  /** Set ANNFixedRadiusTreeSearch. */
  void SetANNFixedRadiusTreeSearch( unsigned int kNearestNeighbors, double errorBound, double squaredRadius );

  /** Set ANNPriorityTreeSearch. */
  void SetANNPriorityTreeSearch( unsigned int kNearestNeighbors, double errorBound );

  /**
   * *** Standard metric stuff: ***
   * 
   */

  /** Initialize the metric. */
  virtual void Initialize( void ) throw ( ExceptionObject );

  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters,
    DerivativeType & Derivative ) const;

  /** Get the value for single valued optimizers. */
  MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType& Value, DerivativeType& Derivative ) const;

  /** Set alpha from alpha - mutual information. */
  itkSetClampMacro( Alpha, double, 0.0, 1.0 );

  /** Get alpha from alpha - mutual information. */
  itkGetConstReferenceMacro( Alpha, double );
  
protected:
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric();
  virtual ~KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Member variables. */
  typename BinaryKNNTreeType::Pointer       m_BinaryKNNTreeFixedIntensity;
  typename BinaryKNNTreeType::Pointer       m_BinaryKNNTreeMovingIntensity;
  typename BinaryKNNTreeType::Pointer       m_BinaryKNNTreeJointIntensity;

  typename BinaryKNNTreeSearchType::Pointer m_BinaryKNNTreeSearcherFixedIntensity;
  typename BinaryKNNTreeSearchType::Pointer m_BinaryKNNTreeSearcherMovingIntensity;
  typename BinaryKNNTreeSearchType::Pointer m_BinaryKNNTreeSearcherJointIntensity;

  double   m_Alpha;

private:
  KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric(const Self&);  //purposely not implemented
  void operator=(const Self&);                                  //purposely not implemented

  unsigned long m_NumberOfParameters;

}; // end class KNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric.txx"
#endif

#endif // end #ifndef __itkKNNGraphMultiDimensionalAlphaMutualInformationImageToImageMetric_h

