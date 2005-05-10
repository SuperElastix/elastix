/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkMutualInformationImageToImageMetricWithMask_txx
#define _itkMutualInformationImageToImageMetricWithMask_txx

#include "itkMutualInformationImageToImageMetricWithMask.h"

#include "itkCovariantVector.h"
//#include "itkImageRandomConstIteratorWithIndex.h"
#include "vnl/vnl_math.h"
#include "itkGaussianKernelFunction.h"

/** elastix random iterator (that behaves the same in linux and windows) */
#include "itkImageMoreRandomConstIteratorWithIndex.h"

namespace itk
{

/*
 * Constructor
 */
template < class TFixedImage, class TMovingImage >
MutualInformationImageToImageMetricWithMask<TFixedImage,TMovingImage>
::MutualInformationImageToImageMetricWithMask()
{

  this->m_NumberOfSpatialSamples = 0;
  this->SetNumberOfSpatialSamples( 50 );

  this->m_KernelFunction  = dynamic_cast<KernelFunction*>(
    GaussianKernelFunction::New().GetPointer() );

  this->m_FixedImageStandardDeviation = 0.4;
  this->m_MovingImageStandardDeviation = 0.4;

  this->m_MinProbability = 0.0001;

  // Following initialization is related to
  // calculating image derivatives
  this->m_ComputeGradient = false; // don't use the default gradient for now
  this->m_DerivativeCalculator = DerivativeFunctionType::New();

}


template < class TFixedImage, class TMovingImage  >
void
MutualInformationImageToImageMetricWithMask<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "NumberOfSpatialSamples: ";
  os << this->m_NumberOfSpatialSamples << std::endl;
  os << indent << "FixedImageStandardDeviation: ";
  os << this->m_FixedImageStandardDeviation << std::endl;
  os << indent << "MovingImageStandardDeviation: ";
  os << this->m_MovingImageStandardDeviation << std::endl;
  os << indent << "KernelFunction: ";
  os << this->m_KernelFunction.GetPointer() << std::endl;
}


/*
 * Set the number of spatial samples
 */
template < class TFixedImage, class TMovingImage  >
void
MutualInformationImageToImageMetricWithMask<TFixedImage,TMovingImage>
::SetNumberOfSpatialSamples( 
  unsigned int num )
{
  if ( num == this->m_NumberOfSpatialSamples ) return;

  this->Modified();
 
  // clamp to minimum of 1
  this->m_NumberOfSpatialSamples = ((num > 1) ? num : 1 );

  // resize the storage vectors
  this->m_SampleA.resize( this->m_NumberOfSpatialSamples );
  this->m_SampleB.resize( this->m_NumberOfSpatialSamples );

}


/*
 * Uniformly sample the fixed image domain. Each sample consists of:
 *  - the fixed image value
 *  - the corresponding moving image value
 */
template < class TFixedImage, class TMovingImage  >
void
MutualInformationImageToImageMetricWithMask<TFixedImage,TMovingImage>
::SampleFixedImageDomain(
  SpatialSampleContainer& samples ) const
{
  //typedef ImageRandomConstIteratorWithIndex<FixedImageType> RandomIterator;
	typedef ImageMoreRandomConstIteratorWithIndex<FixedImageType> RandomIterator;

  RandomIterator randIter( this->m_FixedImage, this->GetFixedImageRegion() );

  randIter.GoToBegin();

  typename SpatialSampleContainer::iterator iter;
  typename SpatialSampleContainer::const_iterator end = samples.end();

  bool allOutside = true;

	/** If no mask.*/
	if ( !(this->m_FixedImageMask) )
	{
		/** Set number of samples equal to m_NumberOfSpatialSamples.*/
		randIter.SetNumberOfSamples( this->m_NumberOfSpatialSamples );

		for( iter = samples.begin(); iter != end; ++iter )
    {
			// get sampled index
			FixedImageIndexType index = randIter.GetIndex();
			
			// get sampled fixed image value
			(*iter).FixedImageValue = randIter.Get();
			
			// get moving image value
			this->m_FixedImage->TransformIndexToPhysicalPoint( index, 
				(*iter).FixedImagePointValue );
			
			MovingImagePointType mappedPoint = 
				this->m_Transform->TransformPoint( (*iter).FixedImagePointValue );
			
			if( this->m_Interpolator->IsInsideBuffer( mappedPoint ) )
      {
				(*iter).MovingImageValue = this->m_Interpolator->Evaluate( mappedPoint );
				allOutside = false;
      }
			else
      {
				(*iter).MovingImageValue = 0;
      }
			
			// jump to random position
			++randIter;
			
    } // end for loop
	} // end if no mask
	else
	{
		/** If there is a mask.*/

		/** Set number of samples equal to m_NumberOfSpatialSamples.*/
		randIter.SetNumberOfSamples( 50 * this->m_NumberOfSpatialSamples );

		for( iter = samples.begin(); iter != end; ++iter )
    {
			/** Start jumping around untill a point within the mask is found.*/
			do
			{
				// jump to random position
				++randIter;

				// get sampled index
				FixedImageIndexType index = randIter.GetIndex();
				
				// get moving image value
				this->m_FixedImage->TransformIndexToPhysicalPoint( index, 
					(*iter).FixedImagePointValue );
				
			} while ( !(this->m_FixedImageMask->IsInside((*iter).FixedImagePointValue)) );

			// get sampled fixed image value
			(*iter).FixedImageValue = randIter.Get();

			// Get the mapped point
			MovingImagePointType mappedPoint = 
				this->m_Transform->TransformPoint( (*iter).FixedImagePointValue );
			
			if( this->m_Interpolator->IsInsideBuffer( mappedPoint ) && this->m_MovingImageMask->IsInside( mappedPoint ) )
			{
				(*iter).MovingImageValue = this->m_Interpolator->Evaluate( mappedPoint );
				allOutside = false;
			}
			else
			{
				(*iter).MovingImageValue = 0;
			}

    } // end for loop
	} // end if there is a mask

  if( allOutside )
    {
    // if all the samples mapped to the outside throw an exception
    itkExceptionMacro(<<"All the sampled point mapped to outside of the moving image" );
    }

}


/*
 * Get the match Measure
 */
template < class TFixedImage, class TMovingImage  >
typename MutualInformationImageToImageMetricWithMask<TFixedImage,TMovingImage>
::MeasureType
MutualInformationImageToImageMetricWithMask<TFixedImage,TMovingImage>
::GetValue( const ParametersType& parameters ) const
{

  // make sure the transform has the current parameters
  this->m_Transform->SetParameters( parameters );

  // collect sample set A
  this->SampleFixedImageDomain( this->m_SampleA );

  // collect sample set B
  this->SampleFixedImageDomain( this->m_SampleB );

  // calculate the mutual information
  double dLogSumFixed = 0.0;
  double dLogSumMoving    = 0.0;
  double dLogSumJoint  = 0.0;

  typename SpatialSampleContainer::const_iterator aiter;
  typename SpatialSampleContainer::const_iterator aend = this->m_SampleA.end();
  typename SpatialSampleContainer::const_iterator biter;
  typename SpatialSampleContainer::const_iterator bend = this->m_SampleB.end();

  for( biter = m_SampleB.begin() ; biter != bend; ++biter )
    {
    double dSumFixed  = this->m_MinProbability;
    double dSumMoving     = this->m_MinProbability;
    double dSumJoint   = this->m_MinProbability;

    for( aiter = this->m_SampleA.begin() ; aiter != aend; ++aiter )
      {
      double valueFixed;
      double valueMoving;

      valueFixed = ( (*biter).FixedImageValue - (*aiter).FixedImageValue ) /
        this->m_FixedImageStandardDeviation;
      valueFixed = this->m_KernelFunction->Evaluate( valueFixed );

      valueMoving = ( (*biter).MovingImageValue - (*aiter).MovingImageValue ) /
        this->m_MovingImageStandardDeviation;
      valueMoving = this->m_KernelFunction->Evaluate( valueMoving );

      dSumFixed += valueFixed;
      dSumMoving    += valueMoving;
      dSumJoint  += valueFixed * valueMoving;

      } // end of sample A loop

    dLogSumFixed -= log( dSumFixed );
    dLogSumMoving    -= log( dSumMoving );
    dLogSumJoint  -= log( dSumJoint );

    } // end of sample B loop

  double nsamp   = double( this->m_NumberOfSpatialSamples );

  double threshold = -0.5 * nsamp * log( this->m_MinProbability );
  if( dLogSumMoving > threshold || dLogSumFixed > threshold ||
      dLogSumJoint > threshold  )
    {
    // at least half the samples in B did not occur within
    // the Parzen window width of samples in A
    itkExceptionMacro(<<"Standard deviation is too small" );
    }

  MeasureType measure = dLogSumFixed + dLogSumMoving - dLogSumJoint;
  measure /= nsamp;
  measure += log( nsamp );

  return measure;

}


/*
 * Get the both Value and Derivative Measure
 */
template < class TFixedImage, class TMovingImage  >
void
MutualInformationImageToImageMetricWithMask<TFixedImage,TMovingImage>
::GetValueAndDerivative(
  const ParametersType& parameters,
  MeasureType& value,
  DerivativeType& derivative) const
{

  value = NumericTraits< MeasureType >::Zero;
  unsigned int numberOfParameters = this->m_Transform->GetNumberOfParameters();
  DerivativeType temp( numberOfParameters );
  temp.Fill( 0 );
  derivative = temp;

  // make sure the transform has the current parameters
  this->m_Transform->SetParameters( parameters );

  // set the DerivativeCalculator
  this->m_DerivativeCalculator->SetInputImage( this->m_MovingImage );

  // collect sample set A
  this->SampleFixedImageDomain( this->m_SampleA );

  // collect sample set B
  this->SampleFixedImageDomain( this->m_SampleB );


  // calculate the mutual information
  double dLogSumFixed = 0.0;
  double dLogSumMoving    = 0.0;
  double dLogSumJoint  = 0.0;

  typename SpatialSampleContainer::iterator aiter;
  typename SpatialSampleContainer::const_iterator aend = this->m_SampleA.end();
  typename SpatialSampleContainer::iterator biter;
  typename SpatialSampleContainer::const_iterator bend = this->m_SampleB.end();

  // precalculate all the image derivatives for sample A
  typedef std::vector<DerivativeType> DerivativeContainer;
  DerivativeContainer sampleADerivatives;
  sampleADerivatives.resize( this->m_NumberOfSpatialSamples );

  typename DerivativeContainer::iterator aditer;
  DerivativeType tempDeriv( numberOfParameters );

  for( aiter = this->m_SampleA.begin(), aditer = sampleADerivatives.begin();
       aiter != aend; ++aiter, ++aditer )
    {
    /*** FIXME: is there a way to avoid the extra copying step? *****/
    this->CalculateDerivatives( (*aiter).FixedImagePointValue, tempDeriv );
    (*aditer) = tempDeriv;
    }


  DerivativeType derivB(numberOfParameters);

  for( biter = this->m_SampleB.begin(); biter != bend; ++biter )
    {
    double dDenominatorMoving = this->m_MinProbability;
    double dDenominatorJoint = this->m_MinProbability;

    double dSumFixed = this->m_MinProbability;

    for( aiter = this->m_SampleA.begin(); aiter != aend; ++aiter )
      {
      double valueFixed;
      double valueMoving;

      valueFixed = ( (*biter).FixedImageValue - (*aiter).FixedImageValue )
        / this->m_FixedImageStandardDeviation;
      valueFixed = this->m_KernelFunction->Evaluate( valueFixed );

      valueMoving = ( (*biter).MovingImageValue - (*aiter).MovingImageValue )
        / this->m_MovingImageStandardDeviation;
      valueMoving = this->m_KernelFunction->Evaluate( valueMoving );

      dDenominatorMoving += valueMoving;
      dDenominatorJoint += valueMoving * valueFixed;

      dSumFixed += valueFixed;

      } // end of sample A loop

    dLogSumFixed -= log( dSumFixed );
    dLogSumMoving    -= log( dDenominatorMoving );
    dLogSumJoint  -= log( dDenominatorJoint );

    // get the image derivative for this B sample
    this->CalculateDerivatives( (*biter).FixedImagePointValue, derivB );

    double totalWeight = 0.0;

    for( aiter = this->m_SampleA.begin(), aditer = sampleADerivatives.begin();
         aiter != aend; ++aiter, ++aditer )
      {
      double valueFixed;
      double valueMoving;
      double weightMoving;
      double weightJoint;
      double weight;

      valueFixed = ( (*biter).FixedImageValue - (*aiter).FixedImageValue ) /
        this->m_FixedImageStandardDeviation;
      valueFixed = this->m_KernelFunction->Evaluate( valueFixed );

      valueMoving = ( (*biter).MovingImageValue - (*aiter).MovingImageValue ) /
        this->m_MovingImageStandardDeviation;
      valueMoving = this->m_KernelFunction->Evaluate( valueMoving );

      weightMoving = valueMoving / dDenominatorMoving;
      weightJoint = valueMoving * valueFixed / dDenominatorJoint;

      weight = ( weightMoving - weightJoint );
      weight *= (*biter).MovingImageValue - (*aiter).MovingImageValue;

      totalWeight += weight;
      derivative -= (*aditer) * weight;

      } // end of sample A loop

    derivative += derivB * totalWeight;

    } // end of sample B loop


  double nsamp    = double( this->m_NumberOfSpatialSamples );

  double threshold = -0.5 * nsamp * log( this->m_MinProbability );
  if( dLogSumMoving > threshold || dLogSumFixed > threshold ||
      dLogSumJoint > threshold  )
    {
    // at least half the samples in B did not occur within
    // the Parzen window width of samples in A
    itkExceptionMacro(<<"Standard deviation is too small" );
    }


  value  = dLogSumFixed + dLogSumMoving - dLogSumJoint;
  value /= nsamp;
  value += log( nsamp );

  derivative  /= nsamp;
  derivative  /= vnl_math_sqr( this->m_MovingImageStandardDeviation );

}


/*
 * Get the match measure derivative
 */
template < class TFixedImage, class TMovingImage  >
void
MutualInformationImageToImageMetricWithMask<TFixedImage,TMovingImage>
::GetDerivative( const ParametersType& parameters, DerivativeType & derivative ) const
{
  MeasureType value;
  // call the combined version
  this->GetValueAndDerivative( parameters, value, derivative );
}


/*
 * Calculate derivatives of the image intensity with respect
 * to the transform parmeters.
 *
 * This should really be done by the mapper.
 *
 * This is a temporary solution until this feature is implemented
 * in the mapper. This solution only works for any transform
 * that support GetJacobian()
 */
template < class TFixedImage, class TMovingImage  >
void
MutualInformationImageToImageMetricWithMask<TFixedImage,TMovingImage>
::CalculateDerivatives(
  const FixedImagePointType& point,
  DerivativeType& derivatives ) const
{

  MovingImagePointType mappedPoint = this->m_Transform->TransformPoint( point );
  
  CovariantVector<double,MovingImageDimension> imageDerivatives;

  if ( this->m_DerivativeCalculator->IsInsideBuffer( mappedPoint ) )
    {
    imageDerivatives = this->m_DerivativeCalculator->Evaluate( mappedPoint );
    }
  else
    {
    derivatives.Fill( 0.0 );
    return;
    }

  typedef typename TransformType::JacobianType JacobianType;
  const JacobianType& jacobian = this->m_Transform->GetJacobian( point );

  unsigned int numberOfParameters = this->m_Transform->GetNumberOfParameters();

  for ( unsigned int k = 0; k < numberOfParameters; k++ )
    {
    derivatives[k] = 0.0;
    for ( unsigned int j = 0; j < MovingImageDimension; j++ )
      {
      derivatives[k] += jacobian[j][k] * imageDerivatives[j];
      }
    } 

}



/*
 * Reinitialize the seed of the random number generator
 */
template < class TFixedImage, class TMovingImage  >
void
MutualInformationImageToImageMetricWithMask<TFixedImage,TMovingImage>
::ReinitializeSeed()
{
  // This method should be the same used in the ImageRandomIterator
  elx_sample_reseed();
}

/*
 * Reinitialize the seed of the random number generator
 */
template < class TFixedImage, class TMovingImage  >
void
MutualInformationImageToImageMetricWithMask<TFixedImage,TMovingImage>
::ReinitializeSeed(int seed)
{
  // This method should be the same used in the ImageRandomIterator
  elx_sample_reseed(seed);
}


} // end namespace itk


#endif

