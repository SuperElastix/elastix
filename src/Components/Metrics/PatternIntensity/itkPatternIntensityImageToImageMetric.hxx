/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkPatternIntensityImageToImageMetric_txx
#define __itkPatternIntensityImageToImageMetric_txx

#include "itkPatternIntensityImageToImageMetric.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkNumericTraits.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>

#include "itkSimpleFilterWatcher.h"
//#include "itkImageFileWriter.h"

namespace itk
{

/**
 * ********************* Constructor ******************************
 */

template <class TFixedImage, class TMovingImage>
PatternIntensityImageToImageMetric<TFixedImage,TMovingImage>
::PatternIntensityImageToImageMetric()
{
  this->m_NormalizationFactor = 1.0;
  this->m_Rescalingfactor = 1.0;
  this->m_DerivativeDelta = 0.001;
  this->m_NoiseConstant = 10000; // = sigma * sigma = 100*100 if not specified
  this->m_NeighborhoodRadius = 3;
  this->m_FixedMeasure = 0;
  this->m_OptimizeNormalizationFactor = false;
  this->m_TransformMovingImageFilter = TransformMovingImageFilterType::New();
  this->m_CombinationTransform = CombinationTransformType::New();
  this->m_RescaleImageFilter = RescaleIntensityImageFilterType::New();
  this->m_DifferenceImageFilter = DifferenceImageFilterType::New();
  this->m_MultiplyByConstantImageFilter = MultiplyByConstantImageFilterType::New();

	} // end Constructor


/**
 * ********************* Initialize ******************************
 */

template <class TFixedImage, class TMovingImage>
void
PatternIntensityImageToImageMetric<TFixedImage,TMovingImage>
::Initialize( void ) throw ( ExceptionObject )
{
  Superclass::Initialize();

  /* resampling for 3D->2D */
  this->m_TransformMovingImageFilter->SetTransform(
    dynamic_cast<RayCastInterpolatorType *>(
    const_cast<  InterpolatorType *>( ( this->GetInterpolator() ) ) )->GetTransform() );
  this->m_TransformMovingImageFilter->SetInterpolator( this->m_Interpolator );
  this->m_TransformMovingImageFilter->SetInput( this->m_MovingImage );
  this->m_TransformMovingImageFilter->SetDefaultPixelValue( 0 );

  this->m_TransformMovingImageFilter->SetSize(
    this->m_FixedImage->GetLargestPossibleRegion().GetSize() );
  this->m_TransformMovingImageFilter->SetOutputOrigin(
    this->m_FixedImage->GetOrigin() );
  this->m_TransformMovingImageFilter->SetOutputSpacing(
    this->m_FixedImage->GetSpacing() );
  this->m_TransformMovingImageFilter->SetOutputDirection(
    this->m_FixedImage->GetDirection() );
  this->m_TransformMovingImageFilter->UpdateLargestPossibleRegion();
  this->ComputeFixedImageExtrema(
    this->GetFixedImage(), this->GetFixedImageRegion() );
  this->ComputeMovingImageExtrema(
    this->m_TransformMovingImageFilter->GetOutput(),
    this->m_TransformMovingImageFilter->GetOutput()->GetBufferedRegion() );
  this->m_NormalizationFactor = this->m_FixedImageTrueMax / this->m_MovingImageTrueMax;
  this->m_MultiplyByConstantImageFilter->SetInput(
    this->m_TransformMovingImageFilter->GetOutput() );
  this->m_MultiplyByConstantImageFilter->SetConstant(
    this->m_NormalizationFactor );
  this->m_DifferenceImageFilter->SetInput1( this->m_FixedImage );
  this->m_DifferenceImageFilter->SetInput2( this->m_MultiplyByConstantImageFilter->GetOutput() );
  this->m_DifferenceImageFilter->UpdateLargestPossibleRegion();
  this->m_FixedMeasure = this->ComputePIFixed();

  /* to rescale the similarity measure between 0-1;*/
  MeasureType tmpmeasure = this->GetValue( this->m_Transform->GetParameters() );

  while ( ( vcl_fabs( tmpmeasure ) / this->m_Rescalingfactor ) > 1 )
	{
    this->m_Rescalingfactor *= 10;
	}

} // end Initialize()


/**
 * ********************* Constructor ******************************PrintSelf
 */

template <class TFixedImage, class TMovingImage>
void
PatternIntensityImageToImageMetric<TFixedImage,TMovingImage>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
    os << indent << "DerivativeDelta: " << this->m_DerivativeDelta << std::endl;

} // end PrintSelf()


/**
 * ********************* ComputePIFixed ******************************
 */

template <class TFixedImage, class TMovingImage>
typename PatternIntensityImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
PatternIntensityImageToImageMetric<TFixedImage,TMovingImage>
::ComputePIFixed( ) const
{
  unsigned int iDimension;
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  MeasureType diff = NumericTraits< MeasureType >::Zero;

  typename FixedImageType::SizeType iterationSize =
    this->m_FixedImage->GetLargestPossibleRegion().GetSize();
  typename FixedImageType::IndexType iterationStartIndex, currentIndex, neighborIndex;
  typename FixedImageType::SizeType neighborIterationSize;
  typename FixedImageType::PointType point;

  for( iDimension = 0; iDimension < this->m_FixedImage->GetImageDimension(); iDimension++ )
  {
    iterationSize[ iDimension ] -= static_cast<int>( 2 * this->m_NeighborhoodRadius );
    iterationStartIndex[ iDimension ] = static_cast<int>( this->m_NeighborhoodRadius );
    neighborIterationSize[ iDimension ] = static_cast<int>(2 * this->m_NeighborhoodRadius) + 1;
	}

  /** replace this by checking the size of the 3rd dimension of fixedimage */
  iterationSize[ 2 ] = neighborIterationSize[ 2 ] = 1;
  iterationStartIndex[ 2 ] = 0;

  typename FixedImageType::RegionType iterationRegion, neighboriterationRegion;
  iterationRegion.SetIndex( iterationStartIndex );
  iterationRegion.SetSize( iterationSize );

  typedef  itk::ImageRegionConstIteratorWithIndex< FixedImageType > FixedImageTypeIteratorType;

  FixedImageTypeIteratorType fixedImageIt( this->m_FixedImage, iterationRegion );
  fixedImageIt.GoToBegin();

  neighboriterationRegion.SetSize( neighborIterationSize );

  bool sampleOK = false;

  if ( this->m_FixedImageMask.IsNull() )
  {
    sampleOK = true;
  }

  while ( !fixedImageIt.IsAtEnd() )
  {
    /** Get current index */
    currentIndex = fixedImageIt.GetIndex();
    this->m_FixedImage->TransformIndexToPhysicalPoint( currentIndex, point );

    /** if fixedMask is given */
    if ( !this->m_FixedImageMask.IsNull() )
    {
      if ( this->m_FixedImageMask->IsInside( point ) )
      {
        sampleOK = true;
      }
      else
      {
        sampleOK = false;
      }
		}

    if ( sampleOK )
    {
      /** setup the neighborhood iterator */
      for( iDimension = 0; iDimension < this->m_FixedImage->GetImageDimension(); iDimension++ )
      {
        neighborIndex[ iDimension ] = currentIndex[ iDimension ] - this->m_NeighborhoodRadius;
        neighborIndex[ 2 ] = 0; // 2D only
      }

    neighboriterationRegion.SetIndex( neighborIndex );
    FixedImageTypeIteratorType neighborIt( this->m_FixedImage, neighboriterationRegion );
    neighborIt.GoToBegin();

      while ( ! neighborIt.IsAtEnd() )
      {
        diff = fixedImageIt.Value() - neighborIt.Value();
        measure += ( this->m_NoiseConstant ) / ( this->m_NoiseConstant + ( diff * diff ) );
        ++neighborIt;
      } // end while neighborIt

    } // end if sampleOK

    ++fixedImageIt;
  } //end while fixedImageIt

  return measure;

} // end ComputePIFixed()


/**
 * ********************* ComputePIDiff ******************************
 */

template <class TFixedImage, class TMovingImage>
typename PatternIntensityImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
PatternIntensityImageToImageMetric<TFixedImage,TMovingImage>
::ComputePIDiff( const TransformParametersType & parameters, float scalingfactor ) const
{
  unsigned int iDimension;
  this->SetTransformParameters( parameters );
  this->m_TransformMovingImageFilter->Modified();
  this->m_MultiplyByConstantImageFilter->SetConstant( scalingfactor );
  this->m_DifferenceImageFilter->UpdateLargestPossibleRegion();
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  MeasureType diff = NumericTraits< MeasureType >::Zero;

  typename FixedImageType::SizeType iterationSize = this->m_FixedImage->GetLargestPossibleRegion().GetSize();
  typename FixedImageType::IndexType iterationStartIndex, currentIndex, neighborIndex;
  typename FixedImageType::SizeType neighborIterationSize;
  typename FixedImageType::PointType point;

  for ( iDimension = 0; iDimension < this->m_FixedImage->GetImageDimension(); iDimension++ )
  {
    iterationSize[iDimension] -= static_cast<int>( 2 * this->m_NeighborhoodRadius );
    iterationStartIndex[iDimension] = static_cast<int>( this->m_NeighborhoodRadius );
    neighborIterationSize[iDimension] = static_cast<int>( 2 * this->m_NeighborhoodRadius + 1 );
  }

  /** 3rd Dimension is 1 */
  iterationSize[ 2 ] = neighborIterationSize[ 2 ] = 1;
  iterationStartIndex[ 2 ] = 0;

  typename FixedImageType::RegionType iterationRegion, neighboriterationRegion;
  iterationRegion.SetIndex( iterationStartIndex );
  iterationRegion.SetSize( iterationSize );

  typedef itk::ImageRegionConstIteratorWithIndex< TransformedMovingImageType > DifferenceImageIteratorType;
  DifferenceImageIteratorType differenceImageIt( this->m_DifferenceImageFilter->GetOutput(), iterationRegion );
  differenceImageIt.GoToBegin();

  neighboriterationRegion.SetSize( neighborIterationSize );

  bool sampleOK = false;

  if ( this->m_FixedImageMask.IsNull() )
  {
    sampleOK = true;
  }

  while ( !differenceImageIt.IsAtEnd() )
  {
    /** Get current index */
    currentIndex = differenceImageIt.GetIndex();
    this->m_FixedImage->TransformIndexToPhysicalPoint( currentIndex, point );

    /** if fixedMask is given */
    if ( !this->m_FixedImageMask.IsNull() )
    {
      if ( this->m_FixedImageMask->IsInside( point ) )
      {
        sampleOK = true;
      }
      else
      {
        sampleOK = false;
      }
    }

    if ( sampleOK )
    {
      /** setup the neighborhood iterator */
      for ( iDimension = 0; iDimension < this->m_FixedImage->GetImageDimension(); iDimension++ )
      {
        neighborIndex[ iDimension ] = currentIndex[ iDimension ] - this->m_NeighborhoodRadius;
        neighborIndex[ 2 ] = 0; // 2D only
      }

      neighboriterationRegion.SetIndex( neighborIndex );
      DifferenceImageIteratorType neighborIt( this->m_DifferenceImageFilter->GetOutput(), neighboriterationRegion );
      neighborIt.GoToBegin();

      while ( !neighborIt.IsAtEnd() )
      {
        diff = differenceImageIt.Value() - neighborIt.Value();
        measure += ( this->m_NoiseConstant )/( this->m_NoiseConstant + ( diff * diff ) );
        ++neighborIt;
      } // end while neighborIt

    } // end if sampleOK

    ++differenceImageIt;
  } // end while differenceImageIt

  return measure;

} // end ComputePIDiff()


/**
 * ********************* GetValue ******************************
 */

template <class TFixedImage, class TMovingImage>
typename PatternIntensityImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
PatternIntensityImageToImageMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
  this->SetTransformParameters( parameters );
  this->m_TransformMovingImageFilter->Modified();
  this->m_DifferenceImageFilter->UpdateLargestPossibleRegion();
  MeasureType measure = 1e10;
  MeasureType currentMeasure = 1e10;

  if ( this->m_OptimizeNormalizationFactor )
  {
    float tmpfactor =  0.0;
    float factorstep =  ( this->m_NormalizationFactor * 10 - tmpfactor ) / 100;
    float bestfactor = tmpfactor;
    MeasureType tmpMeasure = 1e10;

    while ( tmpfactor <=  this->m_NormalizationFactor * 1.0 )
    {
      measure = this->ComputePIDiff( parameters, tmpfactor );
      tmpMeasure = ( measure - this->m_FixedMeasure ) / - this->m_Rescalingfactor;

      if( tmpMeasure < currentMeasure )
      {
        currentMeasure = tmpMeasure;
        bestfactor = tmpfactor;
      }

      tmpfactor += factorstep;
    }
  }
  else
  {
    measure = this->ComputePIDiff( parameters, this->m_NormalizationFactor );
    currentMeasure = -( measure - this->m_FixedMeasure ) / this->m_Rescalingfactor;
  }

	return currentMeasure;

} // end GetValue()


/**
 * ********************* GetDerivative ******************************
 */

template < class TFixedImage, class TMovingImage>
void
PatternIntensityImageToImageMetric<TFixedImage,TMovingImage>
::GetDerivative( const TransformParametersType & parameters,
  DerivativeType & derivative ) const
{
  TransformParametersType testPoint;
  testPoint = parameters;
  const unsigned int numberOfParameters = this->GetNumberOfParameters();
  derivative = DerivativeType( numberOfParameters );

  for ( unsigned int i = 0; i < numberOfParameters; i++ )
  {
    testPoint[ i ] -= this->m_DerivativeDelta / vcl_sqrt( this->m_Scales[ i ] );
    const MeasureType valuep0 = this->GetValue( testPoint );
    testPoint[ i ] += 2 * this->m_DerivativeDelta / vcl_sqrt( this->m_Scales[ i ] );
    const MeasureType valuep1 = this->GetValue( testPoint );
    derivative[ i ] = ( valuep1 - valuep0 ) / ( 2 * this->m_DerivativeDelta / vcl_sqrt( this->m_Scales[ i ] ) );
    testPoint[ i ] = parameters[ i ];
	}

} // end GetDerivative()


/**
 * ********************* GetValueAndDerivative ******************************
 */

template <class TFixedImage, class TMovingImage>
void
PatternIntensityImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative( const TransformParametersType & parameters,
  MeasureType & Value, DerivativeType & derivative ) const
{
  Value = this->GetValue( parameters );
  this->GetDerivative( parameters, derivative );

} // end GetValueAndDerivative()

} // end namespace itk

#endif // end __itkPatternIntensityImageToImageMetric_txx
