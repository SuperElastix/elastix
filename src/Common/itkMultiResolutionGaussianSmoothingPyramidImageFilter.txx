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
#ifndef _itkMultiResolutionGaussianSmoothingPyramidImageFilter_txx
#define _itkMultiResolutionGaussianSmoothingPyramidImageFilter_txx

#include "itkMultiResolutionGaussianSmoothingPyramidImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkRecursiveGaussianImageFilter.h"
#include "itkExceptionObject.h"

#include "vnl/vnl_math.h"

namespace itk
{


/*
 * Constructor
 */
template <class TInputImage, class TOutputImage>
MultiResolutionGaussianSmoothingPyramidImageFilter<TInputImage, TOutputImage>
::MultiResolutionGaussianSmoothingPyramidImageFilter()
{
}


/*
 * Set the multi-resolution schedule
 */
template <class TInputImage, class TOutputImage>
void
MultiResolutionGaussianSmoothingPyramidImageFilter<TInputImage, TOutputImage>
::SetSchedule( const ScheduleType& schedule )
{
  if( schedule.rows() != this->m_NumberOfLevels ||
      schedule.columns() != ImageDimension )
  {
    itkDebugMacro(<< "Schedule has wrong dimensions" );
    return;
  }

  if( schedule == this->m_Schedule )
  {
    return;
  }

  this->Modified();
  unsigned int level, dim;
  for( level = 0; level < this->m_NumberOfLevels; level++ )
  {
    for( dim = 0; dim < ImageDimension; dim++ )
    {

      this->m_Schedule[level][dim] = schedule[level][dim];
      
      /** Minimum schedule of 0. For the rest no restrictions
       * as imposed in the superclass */
      if( this->m_Schedule[level][dim] < 0 )
      {
        this->m_Schedule[level][dim] = 0;
      }
    }
  }
}

/*
 * GenerateData for non downward divisible schedules
 */
template <class TInputImage, class TOutputImage>
void
MultiResolutionGaussianSmoothingPyramidImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  // Get the input and output pointers
  InputImageConstPointer  inputPtr = this->GetInput();

  // Create caster and smoother  filters
  typedef CastImageFilter<InputImageType, OutputImageType> CasterType;
  typedef RecursiveGaussianImageFilter<OutputImageType, OutputImageType> SmootherType;
  typedef typename SmootherType::Pointer SmootherPointer;
  typedef FixedArray<SmootherPointer, ImageDimension>  SmootherArrayType;
  typedef typename InputImageType::SpacingType SpacingType;
  
  typename CasterType::Pointer caster = CasterType::New();
  SmootherArrayType smootherArray;
  for ( unsigned int i = 0; i < ImageDimension; ++i)
  {
    smootherArray[i] = SmootherType::New();
    smootherArray[i]->SetDirection(i);
    smootherArray[i]->SetZeroOrder();
    smootherArray[i]->SetNormalizeAcrossScale(false);
    smootherArray[i]->ReleaseDataFlagOn();
  }
  
  // connect the filters
  caster->SetInput( inputPtr );
  
  smootherArray[0]->SetInput( caster->GetOutput() );
  for ( i = 1; i < ImageDimension; ++i)
  {
    smootherArray[i]->SetInput( smootherArray[i-1]->GetOutput() );
  } 
 
  /** Set the standard deviation and do the smoothing */
  unsigned int ilevel, idim;
  unsigned int factors[ImageDimension];
  double       stdev[ImageDimension];
  SpacingType spacing = inputPtr->GetSpacing();

  for( ilevel = 0; ilevel < this->m_NumberOfLevels; ilevel++ )
  {

    this->UpdateProgress( static_cast<float>( ilevel ) /
                          static_cast<float>( this->m_NumberOfLevels ) );

    // Allocate memory for each output
    OutputImagePointer outputPtr = this->GetOutput( ilevel );
    outputPtr->SetBufferedRegion( outputPtr->GetRequestedRegion() );
    outputPtr->Allocate();

    // compute shrink factors and variances
    for( idim = 0; idim < ImageDimension; idim++ )
    {
      factors[idim] = this->m_Schedule[ilevel][idim];
      /** Compute the standard deviation: 0.5 * factor * spacing
       * This is exactly like in the superclass
       * In the superclass, the DiscreteGaussianImageFilter is used, which
       * requires the variance, and has the option to ignore the image spacing.
       * That's why the formula looks maybe different at first sight.   */
      if ( factors[idim] == 0 )
      {
        stdev[idim]=0.01*spacing[idim];
      }
      else
      {
        stdev[idim] = 0.5 * static_cast<float>( factors[idim] )*spacing[idim];
      }
      smootherArray[idim]->SetSigma( stdev[idim] );
      // force to always update in case shrink factors are the same 
      // (SK: why? is this because we reuse this filter for every resolution?)
      smootherArray[idim]->Modified();
    }
    
    smootherArray[ImageDimension-1]->GraftOutput( outputPtr );
    smootherArray[ImageDimension-1]->Update();
   
    this->GraftNthOutput( ilevel, smootherArray[ImageDimension-1]->GetOutput() );

  } // for ilevel...

}


/*
 * PrintSelf method
 */
template <class TInputImage, class TOutputImage>
void
MultiResolutionGaussianSmoothingPyramidImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}


/* 
 * GenerateOutputInformation
 */
template <class TInputImage, class TOutputImage>
void
MultiResolutionGaussianSmoothingPyramidImageFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
  // call the supersuperclass's implementation of this method
  typedef typename Superclass::Superclass SuperSuperclass;
  SuperSuperclass::GenerateOutputInformation();

  // get pointers to the input and output
  InputImageConstPointer inputPtr = this->GetInput();

  if ( !inputPtr  )
    {
    itkExceptionMacro( << "Input has not been set" );
    }

  OutputImagePointer outputPtr;
   
  unsigned int ilevel;
  for( ilevel = 0; ilevel < this->m_NumberOfLevels; ilevel++ )
  {
    /** The same as the input image for each resolution
     * \todo: is this not already done in the supersuperclass?  */
    OutputImagePointer outputPtr = this->GetOutput( ilevel );
    if( !outputPtr ) { continue; }

    outputPtr->SetLargestPossibleRegion( inputPtr->GetLargestPossibleRegion() );
    outputPtr->SetSpacing( inputPtr->GetSpacing() );
  }

}


/* 
 * GenerateOutputRequestedRegion
 */
template <class TInputImage, class TOutputImage>
void
MultiResolutionGaussianSmoothingPyramidImageFilter<TInputImage, TOutputImage>
::GenerateOutputRequestedRegion(DataObject * refOutput )
{
  // call the supersuperclass's implementation of this method
  typedef typename Superclass::Superclass SuperSuperclass;
  SuperSuperclass::GenerateOutputRequestedRegion( refOutput );

  // find the index for this output
  unsigned int refLevel = refOutput->GetSourceOutputIndex();

  // compute baseIndex and baseSize
  typedef typename OutputImageType::SizeType    SizeType;
  typedef typename SizeType::SizeValueType      SizeValueType;
  typedef typename OutputImageType::IndexType   IndexType;
  typedef typename IndexType::IndexValueType    IndexValueType;
  typedef typename OutputImageType::RegionType  RegionType;

  /** \todo: shouldn't this be a dynamic_cast? */
  TOutputImage * ptr = static_cast<TOutputImage*>( refOutput );
  if( !ptr )
  {
    itkExceptionMacro( << "Could not cast refOutput to TOutputImage*." );
  }

  unsigned int ilevel;

  if ( ptr->GetRequestedRegion() == ptr->GetLargestPossibleRegion() )
  {

    // set the requested regions for the other outputs to their 
    // requested region

    for( ilevel = 0; ilevel < this->m_NumberOfLevels; ilevel++ )
      {
      if( ilevel == refLevel ) { continue; }
      if( !this->GetOutput(ilevel) ) { continue; }
    
      this->GetOutput(ilevel)->SetRequestedRegionToLargestPossibleRegion();      
      }

  }
  else
  {
    // compute requested regions for the other outputs based on
    // the requested region of the reference output

    /** Set them all to the same region */
    RegionType outputRegion = ptr->GetRequestedRegion();

    for( ilevel = 0; ilevel < this->m_NumberOfLevels; ilevel++ )
    {
      if( ilevel == refLevel ) { continue; }
      if( !this->GetOutput(ilevel) ) { continue; }
            
      // make sure the region is within the largest possible region
      outputRegion.Crop( this->GetOutput( ilevel )->
                         GetLargestPossibleRegion() );
      // set the requested region
      this->GetOutput( ilevel )->SetRequestedRegion( outputRegion );
    }
  }

}


/* 
 * GenerateInputRequestedRegion
 */
template <class TInputImage, class TOutputImage>
void
MultiResolutionGaussianSmoothingPyramidImageFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
{
  // call the supersuperclass's implementation of this method. This should
  // copy the output requested region to the input requested region
  typedef typename Superclass::Superclass SuperSuperclass;
  SuperSuperclass::GenerateInputRequestedRegion();
  
  // This filter needs all of the input, because it uses the 
  // the GausianRecursiveFilter.
  InputImagePointer image = const_cast<InputImageType *>( this->GetInput() );

  if ( !image )
  {
    itkExceptionMacro( << "Input has not been set." );
  }

  if( image )
  {
    image->SetRequestedRegion( this->GetInput()->GetLargestPossibleRegion() );
  }

}


/* 
 * EnlargeOutputRequestedRegion
 */

template <class TInputImage, class TOutputImage>
void
MultiResolutionGaussianSmoothingPyramidImageFilter<TInputImage, TOutputImage>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  TOutputImage *out = dynamic_cast<TOutputImage*>(output);

  if (out)
  {
    out->SetRequestedRegion( out->GetLargestPossibleRegion() );
  }
}


} // namespace itk

#endif
