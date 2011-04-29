/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkGradientDifferenceImageToImageMetric2.hxx,v $
  Language:  C++
  Date:      $Date: 2011-29-04 14:33 $
  Version:   $Revision: 2.0 $

  Copyright (c) 2002 Insight Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkGradientDifferenceImageToImageMetric2_txx
#define __itkGradientDifferenceImageToImageMetric2_txx

#include "itkGradientDifferenceImageToImageMetric2.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkNumericTraits.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImageFileWriter.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>

#include "itkSimpleFilterWatcher.h"
namespace itk
{

  /**
   * ********************* Constructor ******************************
   */

  template <class TFixedImage, class TMovingImage>
    GradientDifferenceImageToImageMetric<TFixedImage,TMovingImage>
	::GradientDifferenceImageToImageMetric()
  {

	unsigned int iDimension;
    m_CastMovedImageFilter = CastMovedImageFilterType::New();
    m_CastFixedImageFilter = CastFixedImageFilterType::New();
    m_CombinationTransform = CombinationTransformType::New();
    m_TransformMovingImageFilter = TransformMovingImageFilterType::New();

	for (iDimension=0; iDimension<FixedImageDimension; iDimension++)
    {
      m_MinFixedGradient[iDimension] = 0;
      m_MaxFixedGradient[iDimension] = 0;
      m_Variance[iDimension] = 0;
    }

	for (iDimension=0; iDimension<MovedImageDimension; iDimension++)
    {
      m_MinMovedGradient[iDimension] = 0;
      m_MaxMovedGradient[iDimension] = 0;
    }

	this->m_DerivativeDelta = 0.001;
	this->m_Rescalingfactor = 1.0;
}


  /**
   * ********************* Initialize ******************************
   */

  template <class TFixedImage, class TMovingImage>
    void
	GradientDifferenceImageToImageMetric<TFixedImage,TMovingImage>
	::Initialize(void) throw ( ExceptionObject )
  {
  
	unsigned int iFilter;

	/** Initialise the base class */
	Superclass::Initialize();

	/** Resampling for 3D->2D */
	m_TransformMovingImageFilter->SetTransform( dynamic_cast<CombinationTransformType *>(
	  dynamic_cast<RayCastInterpolatorType *>(
	  const_cast<  InterpolatorType *>( (this->GetInterpolator() ) ) )->GetTransform() ) );
	m_TransformMovingImageFilter->SetInterpolator( this->m_Interpolator );
	m_TransformMovingImageFilter->SetInput( this->m_MovingImage );
	m_TransformMovingImageFilter->SetDefaultPixelValue( 0 );
	m_TransformMovingImageFilter->SetSize( this->m_FixedImage->GetLargestPossibleRegion().GetSize() );
	m_TransformMovingImageFilter->SetOutputOrigin( this->m_FixedImage->GetOrigin() );
	m_TransformMovingImageFilter->SetOutputSpacing( this->m_FixedImage->GetSpacing() );
	m_TransformMovingImageFilter->SetOutputDirection( this->m_FixedImage->GetDirection() );

	/** Compute the gradient of the fixed image */
	m_CastFixedImageFilter->SetInput( this->m_FixedImage );

	for (iFilter=0; iFilter<FixedImageDimension; iFilter++)
    {
      m_FixedSobelOperators[iFilter].SetDirection( iFilter );
      m_FixedSobelOperators[iFilter].CreateDirectional();
      m_FixedSobelFilters[iFilter] = FixedSobelFilter::New();
      m_FixedSobelFilters[iFilter]->OverrideBoundaryCondition( &m_FixedBoundCond );
      m_FixedSobelFilters[iFilter]->SetOperator( m_FixedSobelOperators[iFilter] );
      m_FixedSobelFilters[iFilter]->SetInput( m_CastFixedImageFilter->GetOutput() );
      m_FixedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
    }

	ComputeVariance();

	/** Compute the gradient of the transformed moving image */
	m_CastMovedImageFilter->SetInput( m_TransformMovingImageFilter->GetOutput() );

	for (iFilter=0; iFilter<MovedImageDimension; iFilter++)
    {
      m_MovedSobelOperators[iFilter].SetDirection( iFilter );
      m_MovedSobelOperators[iFilter].CreateDirectional();
	  m_MovedSobelFilters[iFilter] = MovedSobelFilter::New();
      m_MovedSobelFilters[iFilter]->OverrideBoundaryCondition( &m_MovedBoundCond );
      m_MovedSobelFilters[iFilter]->SetOperator( m_MovedSobelOperators[iFilter] );
      m_MovedSobelFilters[iFilter]->SetInput( m_CastMovedImageFilter->GetOutput() );
      m_MovedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
    }

  /* to rescale the similarity measure between 0-1;*/
  MeasureType tmpmeasure = this->GetValue( this->m_CombinationTransform->GetParameters() );

	while ( (fabs(tmpmeasure)/m_Rescalingfactor) > 1 ){

      m_Rescalingfactor*=10;

	}

  }


/**
 * ********************* PrintSelf ******************************
 */

  template <class TFixedImage, class TMovingImage>
    void
    GradientDifferenceImageToImageMetric<TFixedImage,TMovingImage>
	::PrintSelf(std::ostream& os, Indent indent) const
  {
  
	Superclass::PrintSelf( os, indent );
	os << indent << "DerivativeDelta: " << this->m_DerivativeDelta << std::endl;
  
  }

/**
 * ******************** ComputeMovedGradientRange ******************************
 */

  template <class TFixedImage, class TMovingImage>
	void
	GradientDifferenceImageToImageMetric<TFixedImage,TMovingImage>
	::ComputeMovedGradientRange( void ) const
  {
  
	unsigned int iDimension;
	MovedGradientPixelType gradient;

	for (iDimension=0; iDimension<FixedImageDimension; iDimension++)
    {
      typedef itk::ImageRegionConstIteratorWithIndex<
        MovedGradientImageType > IteratorType;

	  IteratorType iterate( m_MovedSobelFilters[iDimension]->GetOutput(),
        this->GetFixedImageRegion() );
	
	  gradient = iterate.Get();

	  m_MinMovedGradient[iDimension] = gradient;
      m_MaxMovedGradient[iDimension] = gradient;

		while ( ! iterate.IsAtEnd() )
		{
          gradient = iterate.Get();

			if (gradient > m_MaxMovedGradient[iDimension])
			{
			  m_MaxMovedGradient[iDimension] = gradient;
			}

			if (gradient < m_MinMovedGradient[iDimension])
			{
			  m_MinMovedGradient[iDimension] = gradient;
			}


		  ++iterate;
		}
	}
  }


  /**
   * ******************** ComputeVariance ******************************
   */
  template <class TFixedImage, class TMovingImage>
	void
	GradientDifferenceImageToImageMetric<TFixedImage,TMovingImage>
	::ComputeVariance( void ) const
  {
  
	unsigned int iDimension;
	unsigned long nPixels;
	FixedGradientPixelType mean[FixedImageDimension];
	FixedGradientPixelType gradient;

	for (iDimension=0; iDimension<FixedImageDimension; iDimension++)
	{

      typedef itk::ImageRegionConstIteratorWithIndex<
        FixedGradientImageType > IteratorType;

      IteratorType iterate( m_FixedSobelFilters[iDimension]->GetOutput(),
        this->GetFixedImageRegion() );

      /** Calculate the mean gradients */

      nPixels =  0;
      gradient = iterate.Get();
      mean[iDimension] = 0;

      m_MinMovedGradient[iDimension] = gradient;
      m_MaxMovedGradient[iDimension] = gradient;

	  typename FixedImageType::IndexType currentIndex;
      typename FixedImageType::PointType point;

      bool sampleOK = false;

		if ( this->m_FixedImageMask.IsNull() )
          sampleOK = true;

		while ( ! iterate.IsAtEnd() )
        {

		  /** Get current index */
		  currentIndex = iterate.GetIndex();
		  this->m_FixedImage->TransformIndexToPhysicalPoint( currentIndex, point );

		  /** if fixedMask is given */
			if ( !this->m_FixedImageMask.IsNull() )
			{
				if ( this->m_FixedImageMask->IsInside( point ) )
				  sampleOK = true;
				else 
				  sampleOK = false;
			}

			if(sampleOK)
			{ 
			  gradient = iterate.Get();
			  mean[iDimension] += gradient;

				if (gradient > m_MaxFixedGradient[iDimension])
				{
				  m_MaxFixedGradient[iDimension] = gradient;
				}

				if (gradient < m_MinFixedGradient[iDimension])
				{
				  m_MinFixedGradient[iDimension] = gradient;
				}

			  nPixels++;

			} // end if sampleOK

		  ++iterate; 
		}// end while iterate

		if (nPixels > 0)
		{
		  mean[iDimension] /= nPixels;
		}

	  /** Calculate the variance */
	  iterate.GoToBegin();
      m_Variance[iDimension] = 0;

		while ( ! iterate.IsAtEnd() )
		{
		  currentIndex = iterate.GetIndex();
		  this->m_FixedImage->TransformIndexToPhysicalPoint( currentIndex, point );

		  /** if fixedMask is given */
			if ( !this->m_FixedImageMask.IsNull() )
			{
				if ( this->m_FixedImageMask->IsInside( point ) )
				  sampleOK = true;
				else  
				  sampleOK = false;
			}

			if(sampleOK)
			{ 
			  gradient = iterate.Get();
			  gradient -= mean[iDimension];
			  m_Variance[iDimension] += gradient*gradient;

			} // end sampleOK

		  ++iterate;
		}

	  m_Variance[iDimension] /= nPixels;
	} // end for iDimension
  }


  /**
   * ******************** ComputeMeasure ******************************
   */
  template <class TFixedImage, class TMovingImage>
	typename GradientDifferenceImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
	GradientDifferenceImageToImageMetric<TFixedImage,TMovingImage>
	::ComputeMeasure( const TransformParametersType & parameters,
    const double *subtractionFactor ) const
{

	unsigned int iDimension;
	this->SetTransformParameters( parameters );
	m_TransformMovingImageFilter->Modified();
	m_TransformMovingImageFilter->UpdateLargestPossibleRegion();
	MeasureType measure = NumericTraits< MeasureType >::Zero;

	typename FixedImageType::IndexType currentIndex;
	typename FixedImageType::PointType point;

	for (iDimension=0; iDimension<FixedImageDimension; iDimension++)
    {
		if (m_Variance[iDimension] == NumericTraits< MovedGradientPixelType >::Zero)
		{
		  continue;
		}
    
	  /** Iterate over the fixed and moving gradient images
       *  calculating the similarity measure
	   */

	  MovedGradientPixelType movedGradient;
      FixedGradientPixelType fixedGradient;
      MovedGradientPixelType diff;

	  typedef  itk::ImageRegionConstIteratorWithIndex< FixedGradientImageType >
        FixedIteratorType;
	  
	  FixedIteratorType fixedIterator( m_FixedSobelFilters[iDimension]->GetOutput(),
        this->GetFixedImageRegion() );

      typedef  itk::ImageRegionConstIteratorWithIndex< MovedGradientImageType >
        MovedIteratorType;

      MovedIteratorType movedIterator( m_MovedSobelFilters[iDimension]->GetOutput(),
        this->GetFixedImageRegion() );

	  m_FixedSobelFilters[iDimension]->UpdateLargestPossibleRegion();
      m_MovedSobelFilters[iDimension]->UpdateLargestPossibleRegion();

      bool sampleOK = false;

		if ( this->m_FixedImageMask.IsNull() )
		  sampleOK = true;

		while ( ! fixedIterator.IsAtEnd() )
		{

		  /** Get current index */

		  currentIndex = fixedIterator.GetIndex();
		  this->m_FixedImage->TransformIndexToPhysicalPoint( currentIndex, point );

		  /** if fixedMask is given */
			if ( !this->m_FixedImageMask.IsNull() )
			{

				if ( this->m_FixedImageMask->IsInside( point ) )	// sample is good
				  sampleOK = true;
				else  // sample no good
				  sampleOK = false;
			}

			if(sampleOK)
			{
			  movedGradient = movedIterator.Get();
			  fixedGradient  = fixedIterator.Get();
			  diff = fixedGradient - subtractionFactor[iDimension]*movedGradient;
			  measure += m_Variance[iDimension] / ( m_Variance[iDimension] + diff * diff );

			} // end if sampleOK

		  ++fixedIterator;
		  ++movedIterator;
		} // end while fixedIterator
    
	} // end for iDimension

	return measure /= -m_Rescalingfactor; //negative for minimization
}


  /**
   * ******************** GetValue ******************************
   */

  template <class TFixedImage, class TMovingImage>
	typename GradientDifferenceImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
	GradientDifferenceImageToImageMetric<TFixedImage,TMovingImage>
	::GetValue( const TransformParametersType & parameters ) const
  {
  
	unsigned int iFilter;                       
	unsigned int iDimension;

	this->SetTransformParameters( parameters );
	m_TransformMovingImageFilter->Modified();
	m_TransformMovingImageFilter->UpdateLargestPossibleRegion();

	/** Update the gradient images */

	for (iFilter=0; iFilter<MovedImageDimension; iFilter++)
    {
      m_MovedSobelFilters[iFilter]->UpdateLargestPossibleRegion();
    }

	/** Compute the range of the moved image gradients */
	this->ComputeMovedGradientRange();

	MovedGradientPixelType subtractionFactor[FixedImageDimension];
	MeasureType currentMeasure;

	for (iDimension=0; iDimension<FixedImageDimension; iDimension++)
    {
      subtractionFactor[iDimension] = m_MaxFixedGradient[iDimension]/m_MaxMovedGradient[iDimension];
    }

	currentMeasure = this->ComputeMeasure( parameters, subtractionFactor );

	return currentMeasure;
}

/**
 * ******************** GetDerivative ******************************
 */

  template < class TFixedImage, class TMovingImage>
	void
	GradientDifferenceImageToImageMetric<TFixedImage,TMovingImage>
	::GetDerivative( const TransformParametersType & parameters,
    DerivativeType & derivative ) const
  {

	TransformParametersType testPoint;
	testPoint = parameters;

	const unsigned int numberOfParameters = this->GetNumberOfParameters();
	derivative = DerivativeType( numberOfParameters );

	for ( unsigned int i=0; i<numberOfParameters; i++ )
    {
      testPoint[i] -= this->m_DerivativeDelta / sqrt(m_Scales[i]);
      const MeasureType valuep0 = this->GetValue( testPoint );
      testPoint[i] += 2* this->m_DerivativeDelta / sqrt(m_Scales[i]);
      const MeasureType valuep1 = this->GetValue( testPoint );
      derivative[i] = (valuep1 - valuep0 ) / ( 2 * this->m_DerivativeDelta / sqrt(m_Scales[i]) );
     testPoint[i] = parameters[i];
    }
  }


/**
 * ******************** GetValueAndDerivative ******************************
 */

  template <class TFixedImage, class TMovingImage>
	void
	GradientDifferenceImageToImageMetric<TFixedImage,TMovingImage>
	::GetValueAndDerivative(const TransformParametersType & parameters,
    MeasureType & Value, DerivativeType  & Derivative) const
  {
  
	Value      = this->GetValue( parameters );
	this->GetDerivative( parameters, Derivative );
  }

} // end namespace itk


#endif
