/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkComputeDisplacementDistribution_h
#define __itkComputeDisplacementDistribution_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"

#include "itkImageGridSampler.h"
#include "itkImageRandomSamplerBase.h"
#include "itkImageRandomCoordinateSampler.h"

namespace itk
{
/**\class ComputeDisplacementDistribution
 * \brief This is a helper class for the automatic parameter estimation of the ASGD optimizer.
 *
 * More specifically this class computes the Jacobian terms related to the automatic
 * parameter estimation for the adaptive stochastic gradient descent optimizer.
 * Details can be found in the SPIE paper
 *
 * [1]  Y.Qiao, B.P.F. Lelieveldt, M.Staring
 * "Fast automatic estimation of the optimization step size for nonrigid image registration,"
 * SPIE Medical Imaging: Image Processing,February, 2014.
 * http://elastix.isi.uu.nl/marius/publications/2014_c_SPIEMI.php
 *
 */

template<class TFixedImage, class TTransform >
class ComputeDisplacementDistribution :
  public ScaledSingleValuedNonLinearOptimizer
{
public:
  /** Standard ITK.*/
  typedef ComputeDisplacementDistribution                Self;
  typedef ScaledSingleValuedNonLinearOptimizer           Superclass;
  typedef SmartPointer<Self>                             Pointer;
  typedef SmartPointer<const Self>                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ComputeDisplacementDistribution,
    ScaledSingleValuedNonLinearOptimizer );

  /** typedef  */
  typedef TFixedImage                         FixedImageType;
  typedef TTransform                          TransformType;
  typedef typename FixedImageType::RegionType FixedImageRegionType;
  typedef Superclass::ParametersType          ParametersType;
  typedef Superclass::DerivativeType          DerivativeType;
  typedef Superclass::ScalesType              ScalesType;

  /** Type for the mask of the fixed image. Only pixels that are "inside"
   * this mask will be considered for the computation of the Jacobian terms.
   */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    TFixedImage::ImageDimension );
  typedef SpatialObject< itkGetStaticConstMacro( FixedImageDimension ) >  FixedImageMaskType;
  typedef typename FixedImageMaskType::Pointer                            FixedImageMaskPointer;
  typedef typename FixedImageMaskType::ConstPointer                       FixedImageMaskConstPointer;
  typedef typename TransformType::NonZeroJacobianIndicesType              NonZeroJacobianIndicesType;


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

   /** The main functions that performs the computation. */
  virtual void ComputeDistributionTerms( const ParametersType &mu,
    double & jacg, double & maxJJ, std::string method );

protected:
  ComputeDisplacementDistribution();
  virtual ~ComputeDisplacementDistribution() {};

  typename FixedImageType::ConstPointer     m_FixedImage;
  FixedImageRegionType                      m_FixedImageRegion;
  FixedImageMaskConstPointer                m_FixedImageMask;
  typename TransformType::Pointer           m_Transform;
  ScaledSingleValuedCostFunction::Pointer   m_CostFunction;
  SizeValueType                             m_NumberOfJacobianMeasurements;

  typedef typename  FixedImageType::IndexType         FixedImageIndexType;
  typedef typename  FixedImageType::PointType         FixedImagePointType;
  typedef typename  TransformType::JacobianType       JacobianType;
  typedef typename  JacobianType::ValueType           JacobianValueType;

  /** Samplers. */
  typedef ImageSamplerBase<FixedImageType>              ImageSamplerBaseType;
  typedef typename ImageSamplerBaseType::Pointer        ImageSamplerBasePointer;
  typedef ImageRandomSamplerBase<FixedImageType>        ImageRandomSamplerBaseType;
  typedef typename ImageRandomSamplerBaseType::Pointer  ImageRandomSamplerBasePointer;

  typedef ImageGridSampler< FixedImageType >            ImageGridSamplerType;
  typedef typename ImageGridSamplerType::Pointer        ImageGridSamplerPointer;
  typedef typename ImageGridSamplerType
    ::ImageSampleContainerType                          ImageSampleContainerType;
  typedef typename ImageSampleContainerType::Pointer    ImageSampleContainerPointer;

  /** Typedefs for support of sparse Jacobians and AdvancedTransforms. */
  typedef JacobianType                                    TransformJacobianType;
  typedef typename TransformType::ScalarType              CoordinateRepresentationType;
  typedef typename TransformType::NumberOfParametersType  NumberOfParametersType;

  /** Sample the fixed image to compute the Jacobian terms. */
  // \todo: note that this is an exact copy of itk::ComputeJacobianTerms
  // in the future it would be better to refactoring this part of the code
  virtual void SampleFixedImageForJacobianTerms(
    ImageSampleContainerPointer & sampleContainer );

private:
  ComputeDisplacementDistribution( const Self& ); // purposely not implemented
  void operator=( const Self& );                  // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkComputeDisplacementDistribution.hxx"
#endif

#endif // end #ifndef __itkComputeDisplacementDistribution_h
