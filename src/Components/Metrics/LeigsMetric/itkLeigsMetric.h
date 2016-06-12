/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/


#ifndef __itkLeigsMetric_h
#define __itkLeigsMetric_h

#include "itkImageRandomCoordinateSampler.h"
#include "itkAdvancedImageToImageMetric.h"

#include "itkArray.h"
#include "itkListSampleCArray.h"
#include "itkBinaryTreeBase.h"
#include "itkBinaryTreeSearchBase.h"

#include "itkANNkDTree.h"
#include "itkANNBruteForceTree.h"

#include "itkANNStandardTreeSearch.h"

using namespace std;

namespace itk
{
    template < class TFixedImage, class TMovingImage >
class LeigsMetric :
    public AdvancedImageToImageMetric< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef LeigsMetric   Self;
  typedef AdvancedImageToImageMetric<
    TFixedImage, TMovingImage >                   Superclass;
  typedef SmartPointer<Self>                      Pointer;
  typedef SmartPointer<const Self>                ConstPointer;

  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  typedef typename FixedImageRegionType::SizeType         FixedImageSizeType;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( LeigsMetric, AdvancedImageToImageMetric );


  /** Typedefs from the superclass. */
  typedef typename
    Superclass::CoordinateRepresentationType              CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformPointer           TransformPointer;
  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass::InterpolatorType           InterpolatorType;
  typedef typename Superclass::InterpolatorPointer        InterpolatorPointer;
  typedef typename Superclass::RealType                   RealType;
  typedef typename Superclass::GradientPixelType          GradientPixelType;
  typedef typename Superclass::GradientImageType          GradientImageType;
  typedef typename Superclass::GradientImagePointer       GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType    GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass::MovingImageRegionType      MovingImageRegionType;
  typedef typename Superclass::ImageSamplerType           ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer        ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType   ImageSampleContainerType;
  typedef typename
    Superclass::ImageSampleContainerPointer               ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageLimiterType      FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType     MovingImageLimiterType;
  typedef typename
    Superclass::FixedImageLimiterOutputType               FixedImageLimiterOutputType;
 typedef typename
    Superclass::MovingImageLimiterOutputType              MovingImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageDerivativeScalesType           MovingImageDerivativeScalesType;
    typedef typename Superclass::ThreaderType                    ThreaderType;
    typedef typename Superclass::ThreadInfoType                  ThreadInfoType;

    typedef Array< double >                           MeasurementVectorType;
    typedef typename MeasurementVectorType::ValueType MeasurementVectorValueType;
    typedef typename Statistics::ListSampleCArray<
    MeasurementVectorType, double >                   ListSampleType;
    typedef typename ListSampleType::Pointer ListSamplePointer;
    
    typedef BinaryTreeBase< ListSampleType >    BinaryKNNTreeType;
    typedef typename BinaryKNNTreeType::Pointer BinaryKNNTreePointer;
    typedef BinaryTreeSearchBase< ListSampleType >     BinaryKNNTreeSearchType;
    typedef typename BinaryKNNTreeSearchType::Pointer  BinaryKNNTreeSearchPointer;

    typedef ANNkDTree< ListSampleType >                 ANNkDTreeType;
    typedef ANNBruteForceTree< ListSampleType >         ANNBruteForceTreeType;
    typedef ANNStandardTreeSearch< ListSampleType >     ANNStandardTreeSearchType;

    typedef typename BinaryKNNTreeSearchType::IndexArrayType    IndexArrayType;
    typedef typename BinaryKNNTreeSearchType::DistanceArrayType DistanceArrayType;

  /** Get the value for single valued optimizers. */
  virtual MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get the derivatives of the match measure. */
  virtual void GetDerivative(const TransformParametersType & parameters,
    DerivativeType & derivative ) const;

  /** Get value and derivatives for multiple valued optimizers. */
  virtual void GetValueAndDerivative(const TransformParametersType& parameters,
      MeasureType& Value, DerivativeType& Derivative) const;

  /** Get value and derivatives for multiple valued optimizers. */
  virtual void GetValueAndDerivativeSingleThreaded(const TransformParametersType& parameters, MeasureType& Value, DerivativeType& Derivative) const;

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   * \li Call the superclass' implementation.   */

  virtual void Initialize(void) throw ( ExceptionObject );

    itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );
    
    itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );
    
    itkStaticConstMacro( ReducedFixedImageDimension, unsigned int, FixedImageType::ImageDimension - 1 );
    
    itkStaticConstMacro( ReducedMovingImageDimension, unsigned int, MovingImageType::ImageDimension - 1 );

    itkSetMacro( NumSamplesLastDimension, unsigned int );
    
    itkSetMacro( NearestNeighbours, unsigned int );
    
    itkSetMacro( Time, double );
    
    itkSetMacro( SampleLastDimensionRandomly, bool );

    itkSetMacro( ReducedDimensionIndex, unsigned int );

    itkSetMacro( SubtractMean, bool );
    
    itkSetMacro( TransformIsStackTransform, bool );
    
    itkSetMacro( GridSize, FixedImageSizeType );
    
    itkSetMacro( BinaryKNNTree, BinaryKNNTreePointer );

    itkSetMacro( BinaryKNNTreeSearcher, BinaryKNNTreeSearchPointer );

protected:
  LeigsMetric();
  ~LeigsMetric();
  void PrintSelf( std::ostream& os, Indent indent ) const;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  typedef typename Superclass::FixedImageIndexType                FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType           FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType               MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType                FixedImagePointType;
  typedef typename itk::ContinuousIndex< CoordinateRepresentationType, FixedImageDimension >
                                                                  FixedImageContinuousIndexType;
  typedef typename Superclass::MovingImagePointType               MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType     MovingImageContinuousIndexType;
  typedef typename Superclass::BSplineInterpolatorType            BSplineInterpolatorType;
  typedef typename Superclass::CentralDifferenceGradientFilterType CentralDifferenceGradientFilterType;
  typedef typename Superclass::MovingImageDerivativeType          MovingImageDerivativeType;
  typedef typename Superclass::NonZeroJacobianIndicesType         NonZeroJacobianIndicesType;

  /** Computes the innerproduct of transform Jacobian with moving image gradient.
   * The results are stored in imageJacobian, which is supposed
   * to have the right size (same length as Jacobian's number of columns). */
  void EvaluateTransformJacobianInnerProduct(
    const TransformJacobianType & jacobian,
    const MovingImageDerivativeType & movingImageDerivative,
    DerivativeType & imageJacobian) const;

  void ComputeListSamplesLastDim( const ListSamplePointer & listSamplesLastDim ) const;
  void ComputeListSamplesLastDimDerivative( const ListSamplePointer & listSamplesLastDim, std::vector< FixedImagePointType > * fixedImagePointList, std::vector<vnl_vector<double> > * movingImageValueList) const;
  void SampleRandom( const int n, const int m, std::vector< int > & numbers ) const;
    
    void SetANNStandardTreeSearch( unsigned int kNearestNeighbors, double treeError);
    
    void SetANNkDTree( );
    void SetANNBruteForceTree( );

    virtual void InitializeThreadingParameters( void ) const;

    struct LeigsMetricMultiThreaderParameterType
    {
        Self * m_Metric;
    };
    
        LeigsMetricMultiThreaderParameterType m_LeigsMetricThreaderParameters;

    struct LeigsMetricComputeDerivativePerThreadStruct
    {
        DerivativeType st_Derivative;
    };
    

    itkPadStruct( ITK_CACHE_LINE_ALIGNMENT, LeigsMetricComputeDerivativePerThreadStruct,
                 PaddedLeigsMetricComputeDerivativePerThreadStruct );
    
    itkAlignedTypedef( ITK_CACHE_LINE_ALIGNMENT,
                      PaddedLeigsMetricComputeDerivativePerThreadStruct,
                      AlignedLeigsMetricComputeDerivativePerThreadStruct );
    
    mutable AlignedLeigsMetricComputeDerivativePerThreadStruct * m_LeigsMetricComputeDerivativePerThreadVariables;
    mutable ThreadIdType  m_LeigsMetricComputeDerivativePerThreadVariablesSize;
    
    void LaunchComputeDerivativeThreaderCallback( void ) const;
    static ITK_THREAD_RETURN_TYPE ComputeDerivativeThreaderCallback( void * arg );

    void ThreadedComputeDerivative( ThreadIdType threadID );
    void AfterThreadedComputeDerivative( DerivativeType & derivative ) const;

private:
  LeigsMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  mutable std::vector<int> m_RandomList;

  /** Variables to control random sampling in last dimension. */
  unsigned int m_ReducedDimensionIndex;

  unsigned int m_NumSamplesLastDimension;
  bool m_SampleLastDimensionRandomly;
    
  unsigned int m_NearestNeighbours;

  double m_Time;

  /** Bool to determine if we want to subtract the mean derivate from the derivative elements. */
  bool m_SubtractMean;

  /** GridSize of B-spline transform. */
  FixedImageSizeType m_GridSize;

  /** Bool to indicate if the transform used is a stacktransform. Set by elx files. */
  bool m_TransformIsStackTransform;
    
  BinaryKNNTreePointer m_BinaryKNNTree;
  BinaryKNNTreeSearchPointer m_BinaryKNNTreeSearcher;
    
  mutable vnl_sparse_matrix<double>* m_WeightMatrix;
  mutable std::vector< FixedImagePointType >* m_FixedImagePointList;
  mutable std::vector<vnl_vector<double > >* m_MovingImageValueList;
  mutable float m_EigenValue;
  mutable vnl_vector<double> m_EigenVector;

}; // end class LeigsMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLeigsMetric.hxx"
#endif

#endif // end #ifndef __itkLeigsMetric_h
