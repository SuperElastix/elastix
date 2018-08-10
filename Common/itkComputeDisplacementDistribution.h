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
#ifndef __itkComputeDisplacementDistribution_h
#define __itkComputeDisplacementDistribution_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"

#include "itkImageGridSampler.h"
#include "itkImageRandomSamplerBase.h"
#include "itkImageRandomCoordinateSampler.h"
#include "itkImageFullSampler.h"
#include "itkMultiThreader.h"

namespace itk
{
/**\class ComputeDisplacementDistribution
 * \brief This is a helper class for the automatic parameter estimation of the ASGD optimizer.
 *
 * More specifically this class computes the Jacobian terms related to the automatic
 * parameter estimation for the adaptive stochastic gradient descent optimizer.
 * Details can be found in the TMI paper
 *
 * [1] Y. Qiao, B. van Lew, B.P.F. Lelieveldt and M. Staring
 * "Fast Automatic Step Size Estimation for Gradient Descent Optimization of Image Registration,"
 * IEEE Transactions on Medical Imaging, vol. 35, no. 2, pp. 391 - 403, February 2016.
 * http://elastix.isi.uu.nl/marius/publications/2016_j_TMIa.php
 *
 */

template< class TFixedImage, class TTransform >
class ComputeDisplacementDistribution :
  public ScaledSingleValuedNonLinearOptimizer
{
public:

  /** Standard ITK.*/
  typedef ComputeDisplacementDistribution      Self;
  typedef ScaledSingleValuedNonLinearOptimizer Superclass;
  typedef SmartPointer< Self >                 Pointer;
  typedef SmartPointer< const Self >           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ComputeDisplacementDistribution,
    ScaledSingleValuedNonLinearOptimizer );

  /** typedef  */
  typedef TFixedImage                         FixedImageType;
  typedef typename FixedImageType::PixelType  FixedImagePixelType;
  typedef TTransform                          TransformType;
  typedef typename TransformType::Pointer     TransformPointer;
  typedef typename FixedImageType::RegionType FixedImageRegionType;
  typedef Superclass::ParametersType          ParametersType;
  typedef Superclass::DerivativeType          DerivativeType;
  typedef Superclass::ScalesType              ScalesType;

  /** Type for the mask of the fixed image. Only pixels that are "inside"
   * this mask will be considered for the computation of the Jacobian terms.
   */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    TFixedImage::ImageDimension );
  typedef SpatialObject< itkGetStaticConstMacro( FixedImageDimension ) > FixedImageMaskType;
  typedef typename FixedImageMaskType::Pointer                           FixedImageMaskPointer;
  typedef typename FixedImageMaskType::ConstPointer                      FixedImageMaskConstPointer;
  typedef typename TransformType::NonZeroJacobianIndicesType             NonZeroJacobianIndicesType;

  /** Set the fixed image. */
  itkSetConstObjectMacro( FixedImage, FixedImageType );

  /** Set the transform. */
  itkSetObjectMacro( Transform, TransformType );

  /** Set/Get the fixed image mask. */
  itkSetObjectMacro( FixedImageMask, FixedImageMaskType );
  itkSetConstObjectMacro( FixedImageMask, FixedImageMaskType );
  itkGetConstObjectMacro( FixedImageMask, FixedImageMaskType );

  /** Set some parameters. */
  itkSetMacro( NumberOfJacobianMeasurements, SizeValueType );

  /** Set the region over which the metric will be computed. */
  void SetFixedImageRegion( const FixedImageRegionType & region )
  {
    if( region != this->m_FixedImageRegion )
    {
      this->m_FixedImageRegion = region;
    }
  }


  /** Get the region over which the metric will be computed. */
  itkGetConstReferenceMacro( FixedImageRegion, FixedImageRegionType );

  /** The main function that performs the multi-threaded computation. */
  virtual void Compute( const ParametersType & mu,
    double & jacg, double & maxJJ, std::string method );

  /** The main function that performs the single-threaded computation. */
  virtual void ComputeSingleThreaded( const ParametersType & mu,
    double & jacg, double & maxJJ, std::string method );

  virtual void ComputeUsingSearchDirection( const ParametersType & mu,
    double & jacg, double & maxJJ, std::string methods );

  /** Set the number of threads. */
  void SetNumberOfThreads( ThreadIdType numberOfThreads )
  {
    this->m_Threader->SetNumberOfThreads( numberOfThreads );
  }


  virtual void BeforeThreadedCompute( const ParametersType & mu );

  virtual void AfterThreadedCompute( double & jacg, double & maxJJ );

protected:

  ComputeDisplacementDistribution();
  virtual ~ComputeDisplacementDistribution();

  /** Typedefs for multi-threading. */
  typedef itk::MultiThreader             ThreaderType;
  typedef ThreaderType::ThreadInfoStruct ThreadInfoType;

  typename FixedImageType::ConstPointer   m_FixedImage;
  FixedImageRegionType                    m_FixedImageRegion;
  FixedImageMaskConstPointer              m_FixedImageMask;
  TransformPointer                        m_Transform;
  ScaledSingleValuedCostFunction::Pointer m_CostFunction;
  SizeValueType                           m_NumberOfJacobianMeasurements;
  DerivativeType                          m_ExactGradient;
  SizeValueType                           m_NumberOfParameters;
  ThreaderType::Pointer                   m_Threader;

  typedef typename  FixedImageType::IndexType   FixedImageIndexType;
  typedef typename  FixedImageType::PointType   FixedImagePointType;
  typedef typename  TransformType::JacobianType JacobianType;
  typedef typename  JacobianType::ValueType     JacobianValueType;

  /** Samplers. */
  typedef ImageSamplerBase< FixedImageType >     ImageSamplerBaseType;
  typedef typename ImageSamplerBaseType::Pointer ImageSamplerBasePointer;

  typedef ImageFullSampler< FixedImageType >     ImageFullSamplerType;
  typedef typename ImageFullSamplerType::Pointer ImageFullSamplerPointer;

  typedef ImageRandomSamplerBase< FixedImageType >     ImageRandomSamplerBaseType;
  typedef typename ImageRandomSamplerBaseType::Pointer ImageRandomSamplerBasePointer;

  typedef ImageGridSampler< FixedImageType >     ImageGridSamplerType;
  typedef typename ImageGridSamplerType::Pointer ImageGridSamplerPointer;
  typedef typename ImageGridSamplerType
    ::ImageSampleContainerType                   ImageSampleContainerType;
  typedef typename ImageSampleContainerType::Pointer ImageSampleContainerPointer;

  /** Typedefs for support of sparse Jacobians and AdvancedTransforms. */
  typedef JacobianType                                   TransformJacobianType;
  typedef typename TransformType::ScalarType             CoordinateRepresentationType;
  typedef typename TransformType::NumberOfParametersType NumberOfParametersType;

  /** Sample the fixed image to compute the Jacobian terms. */
  // \todo: note that this is an exact copy of itk::ComputeJacobianTerms
  // in the future it would be better to refactoring this part of the code
  virtual void SampleFixedImageForJacobianTerms(
    ImageSampleContainerPointer & sampleContainer );

  /** Launch MultiThread Compute. */
  void LaunchComputeThreaderCallback( void ) const;

  /** Compute threader callback function. */
  static ITK_THREAD_RETURN_TYPE ComputeThreaderCallback( void * arg );

  /** The threaded implementation of Compute(). */
  virtual inline void ThreadedCompute( ThreadIdType threadID );

  /** Initialize some multi-threading related parameters. */
  virtual void InitializeThreadingParameters( void );

  /** To give the threads access to all member variables and functions. */
  struct MultiThreaderParameterType
  {
    Self * st_Self;
  };
  mutable MultiThreaderParameterType m_ThreaderParameters;

  struct ComputePerThreadStruct
  {
    /**  Used for accumulating variables. */
    double        st_MaxJJ;
    double        st_Displacement;
    double        st_DisplacementSquared;
    SizeValueType st_NumberOfPixelsCounted;
  };
  itkPadStruct( ITK_CACHE_LINE_ALIGNMENT, ComputePerThreadStruct,
    PaddedComputePerThreadStruct );
  itkAlignedTypedef( ITK_CACHE_LINE_ALIGNMENT, PaddedComputePerThreadStruct,
    AlignedComputePerThreadStruct );
  mutable AlignedComputePerThreadStruct * m_ComputePerThreadVariables;
  mutable ThreadIdType                    m_ComputePerThreadVariablesSize;

  SizeValueType               m_NumberOfPixelsCounted;
  bool                        m_UseMultiThread;
  ImageSampleContainerPointer m_SampleContainer;

private:

  ComputeDisplacementDistribution( const Self & ); // purposely not implemented
  void operator=( const Self & );                  // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkComputeDisplacementDistribution.hxx"
#endif

#endif // end #ifndef __itkComputeDisplacementDistribution_h
