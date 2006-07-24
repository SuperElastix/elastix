#ifndef __itkKNNGraphAlphaMutualInformationImageToImageMetric_h
#define __itkKNNGraphAlphaMutualInformationImageToImageMetric_h

#include "itkImageToImageMetricWithSampling.h"

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
 * \class KNNGraphAlphaMutualInformationImageToImageMetric
 *
 * \brief Computes similarity between two images to be registered
 *
 * 
 * \ingroup RegistrationMetrics
 */
  
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT KNNGraphAlphaMutualInformationImageToImageMetric :
  public ImageToImageMetricWithSampling< TFixedImage, TMovingImage>
{
public:

  /** Standard itk. */
  typedef KNNGraphAlphaMutualInformationImageToImageMetric   Self;
  typedef ImageToImageMetricWithSampling<
    TFixedImage, TMovingImage >                         Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );
 
  /** Run-time type information (and related methods). */
  itkTypeMacro( KNNGraphAlphaMutualInformationImageToImageMetric, ImageToImageMetricWithSampling );
 
  /** Types transferred from the Superclass. */
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
  typedef typename Superclass::InputPointType			      InputPointType;
  typedef typename Superclass::OutputPointType		      OutputPointType;
  typedef typename Superclass::ImageSamplerType         ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer      ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType      ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer   ImageSampleContainerPointer;

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

  /** Set ANNbdTree. */
  void SetANNbdTree( unsigned int bucketSize, std::string splittingRule, std::string shrinkingRule );

  /** Set ANNBruteForceTree. */
  void SetANNBruteForceTree( void );

  /** Get the binary 1D kNN tree. *
  itkGetConstObjectMacro( BinaryKNNTree1D, BinaryKNNTreeType1D );

  /** Get the binary 2D kNN tree. *
  itkGetConstObjectMacro( BinaryKNNTree2D, BinaryKNNTreeType2D );

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

  /** Get the binary 1D kNN tree searcher. *
  itkGetConstObjectMacro( BinaryKNNTreeSearcher1D, BinaryKNNTreeSearchType1D );

  /** Get the binary 2D kNN tree searcher. *
  itkGetConstObjectMacro( BinaryKNNTreeSearcher2D, BinaryKNNTreeSearchType2D );

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

  /** Provide API to reinitialize the seed of the random number generator. *
  static void ReinitializeSeed();
  static void ReinitializeSeed(int);*/

  /** Set alpha from alpha - mutual information. */
  itkSetClampMacro( Alpha, double, 0.0, 1.0 );

  /** Get alpha from alpha - mutual information. */
  itkGetConstReferenceMacro( Alpha, double );
  
protected:
  KNNGraphAlphaMutualInformationImageToImageMetric();
  virtual ~KNNGraphAlphaMutualInformationImageToImageMetric() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

  /** Member variables. */
  typename BinaryKNNTreeType::Pointer       m_BinaryKNNTreeFixedIntensity;
  typename BinaryKNNTreeType::Pointer       m_BinaryKNNTreeMovingIntensity;
  typename BinaryKNNTreeType::Pointer       m_BinaryKNNTreeJointIntensity;

  typename BinaryKNNTreeSearchType::Pointer m_BinaryKNNTreeSearcherFixedIntensity;
  typename BinaryKNNTreeSearchType::Pointer m_BinaryKNNTreeSearcherMovingIntensity;
  typename BinaryKNNTreeSearchType::Pointer m_BinaryKNNTreeSearcherJointIntensity;

  double    m_Alpha;

private:
  KNNGraphAlphaMutualInformationImageToImageMetric(const Self&);  //purposely not implemented
  void operator=(const Self&);                                  //purposely not implemented

  unsigned long m_NumberOfParameters;

}; // end class KNNGraphAlphaMutualInformationImageToImageMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkKNNGraphAlphaMutualInformationImageToImageMetric.txx"
#endif

#endif // end #ifndef __itkKNNGraphAlphaMutualInformationImageToImageMetric_h

