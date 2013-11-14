/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkComputeDisplacementDistribution_hxx
#define __itkComputeDisplacementDistribution_hxx

#include "itkComputeDisplacementDistribution.h"

#include <string>
#include "vnl/vnl_math.h"
#include "vnl/vnl_fastops.h"
#include "vnl/vnl_diag_matrix.h"

namespace itk
{
/**
 * ************************* Constructor ************************
 */

template<class TFixedImage, class TTransform >
ComputeDisplacementDistribution< TFixedImage,TTransform >
::ComputeDisplacementDistribution()
{
  this->m_FixedImage = NULL;
  this->m_FixedImageMask = NULL;
  this->m_Transform = NULL;
  this->m_FixedImageMask = NULL;
  this->m_NumberOfJacobianMeasurements = 0;

} // end Constructor


/**
 * ************************* ComputeParameters ************************
 */

template<class TFixedImage, class TTransform >
void
ComputeDisplacementDistribution< TFixedImage,TTransform >
::ComputeDistributionTerms( const ParametersType & mu, 
                           double & jacg, double & maxJJ, std::string methods )
{
  /** This function computes four terms needed for the automatic parameter
    * estimation using voxel displacement distribution estimation method.
    * The equation number refers to the SPIE paper.
    * Term 1: jacg = mean( J_j * g ) + var( J_j * g ).
    */

  /** Initialize. */
  maxJJ = jacg = 0.0;

  /** Get samples. */
  ImageSampleContainerPointer sampleContainer = 0;
  this->SampleFixedImageForJacobianTerms( sampleContainer );
  const SizeValueType nrofsamples = sampleContainer->Size();

  /** Get the number of parameters. */
  const unsigned int P = static_cast<unsigned int>(
    this->m_Transform->GetNumberOfParameters() );

  /** Get scales vector */
  const ScalesType & scales = this->GetScales();
  this->m_ScaledCostFunction->SetScales( scales );

  /** Get the exact gradient. */
  DerivativeType exactgradient( P );
  this->GetScaledDerivative( mu, exactgradient );

  /** Get transform and set current position. */
  typename TransformType::Pointer transform = this->m_Transform;
  const unsigned int outdim = this->m_Transform->GetOutputSpaceDimension();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();
  unsigned int samplenr = 0;

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const SizeValueType sizejacind
    = this->m_Transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType jacj( outdim, sizejacind );
  jacj.Fill( 0.0 );
  NonZeroJacobianIndicesType jacind( sizejacind );
  jacind[ 0 ] = 0;
  if ( sizejacind > 1 ) jacind[ 1 ] = 0;

  /**
   * Compute maxJJ and jac*gradient
   */
  DerivativeType Jgg( outdim );
  Jgg.Fill( 0.0 );
  std::vector< double > JGG_k;
  double sum_jacg = 0.0;
  const double sqrt2 = vcl_sqrt( static_cast<double>( 2.0 ) );
  JacobianType jacjjacj( outdim, outdim );

  samplenr = 0;
  for( iter = begin; iter != end; ++iter )
  {
    /** Read fixed coordinates and get Jacobian. */
    const FixedImagePointType & point = (*iter).Value().m_ImageCoordinates;
    this->m_Transform->GetJacobian( point, jacj, jacind  );

    /** Apply scales, if necessary. */
    if( this->GetUseScales() )
    {
      for( unsigned int pi = 0; pi < sizejacind; ++pi )
      {
        const unsigned int p = jacind[ pi ];
        jacj.scale_column( pi, 1.0 / scales[ p ] );
      }
    }

    /** Compute 1st part of JJ: ||J_j||_F^2. */
    double JJ_j = vnl_math_sqr( jacj.frobenius_norm() );

    /** Compute 2nd part of JJ: 2\sqrt{2} || J_j J_j^T ||_F. */
    vnl_fastops::ABt( jacjjacj, jacj, jacj );
    JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

    /** Max_j [JJ_j]. */
    maxJJ = vnl_math_max( maxJJ, JJ_j );

    /** Compute the matrix of jac*gradient */
    for( unsigned int i = 0; i < outdim; ++i )
    {
      double temp = 0.0;
      for( unsigned int j = 0; j <sizejacind; ++j )
      {
        int pj = jacind[ j ];
        temp += jacj( i, j ) * exactgradient( pj );
      }
      Jgg( i ) = temp;
    }

    sum_jacg += Jgg.magnitude();
    JGG_k.push_back( Jgg.magnitude() );
    ++samplenr;

  } // end loop over sample container

  if( methods == "95percentile" )
  {
    /** Compute the 95% percentile of the distribution of JGG_k */
    unsigned int d = static_cast<unsigned int>( nrofsamples * 0.95 );
    std::sort( JGG_k.begin(), JGG_k.end() );
    jacg = ( JGG_k[ d - 1 ] + JGG_k[ d ] + JGG_k[ d + 1 ] ) / 3.0;
  }
  else if( methods == "2sigma" )
  {
    /** Compute the sigma of the distribution of JGG_k. */
    double sigma = 0.0;
    double mean_JGG = sum_jacg /samplenr;
    for( unsigned int i=0; i < nrofsamples-1; ++i )
    {
      sigma += vnl_math_sqr( JGG_k[i] - mean_JGG );
    }
    sigma /= ( nrofsamples - 1 );// unbiased estimation
    jacg = mean_JGG + 2.0 * vcl_sqrt( sigma );
  }
} // end ComputeDistributionTerms()


/**
 * ************************* SampleFixedImageForJacobianTerms ************************
 */

template<class TFixedImage, class TTransform >
void
ComputeDisplacementDistribution< TFixedImage,TTransform >
::SampleFixedImageForJacobianTerms(
  ImageSampleContainerPointer & sampleContainer )
{
  /** Set up grid sampler. */
  ImageGridSamplerPointer sampler = ImageGridSamplerType::New();
  sampler->SetInput( this->m_FixedImage );
  sampler->SetInputImageRegion( this->GetFixedImageRegion() );
  sampler->SetMask( this->m_FixedImageMask );

  /** Determine grid spacing of sampler such that the desired
   * NumberOfJacobianMeasurements is achieved approximately.
   * Note that the actually obtained number of samples may be lower, due to masks.
   * This is taken into account at the end of this function.
   */
  SizeValueType nrofsamples = this->m_NumberOfJacobianMeasurements;
  sampler->SetNumberOfSamples( nrofsamples );

  /** Get samples and check the actually obtained number of samples. */
  sampler->Update();
  sampleContainer = sampler->GetOutput();
  nrofsamples = sampleContainer->Size();

  if( nrofsamples == 0 )
  {
    itkExceptionMacro(
      << "No valid voxels (0/" << this->m_NumberOfJacobianMeasurements
      << ") found to estimate the AdaptiveStochasticGradientDescent parameters." );
  }

} // end SampleFixedImageForJacobianTerms()


} // end namespace itk

#endif // end #ifndef __itkComputeDisplacementDistribution_hxx
