/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef _itkUpsampleBSplineParametersFilter_txx
#define _itkUpsampleBSplineParametersFilter_txx

#include "itkUpsampleBSplineParametersFilter.h"

#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineDecompositionImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkImageRegionConstIterator.h"


namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TArray, class TImage >
UpsampleBSplineParametersFilter<TArray,TImage>
::UpsampleBSplineParametersFilter()
{
  this->m_BSplineOrder = 3;
  
  // Initialize grid settings.
  this->m_CurrentGridOrigin.Fill(0.0);
  this->m_CurrentGridSpacing.Fill(0.0);
  this->m_CurrentGridDirection.Fill(0.0);
  this->m_RequiredGridOrigin.Fill(0.0);
  this->m_RequiredGridSpacing.Fill(0.0);
  this->m_RequiredGridDirection.Fill(0.0);
} // end Constructor()


/**
 * ******************* UpsampleParameters *******************
 */

template< class TArray, class TImage >
void
UpsampleBSplineParametersFilter<TArray,TImage>
::UpsampleParameters( const ArrayType & parameters_in,
  ArrayType & parameters_out )
{
  /** Determine if upsampling is required. */
  if ( !this->DoUpsampling() )
  {
    parameters_out = parameters_in;
    return;
  }

  /** Typedefs. */
  typedef itk::ResampleImageFilter<
    ImageType, ImageType >                        UpsampleFilterType;
  typedef itk::BSplineResampleImageFunction<
    ImageType, ValueType >                        CoefficientUpsampleFunctionType;
  typedef itk::BSplineDecompositionImageFilter<
    ImageType, ImageType >                        DecompositionFilterType;
  typedef ImageRegionConstIterator< ImageType >   IteratorType;

  /** Get the pointer to the data of the input parameters. */
  PixelType * inputDataPointer
    = const_cast<PixelType *>( parameters_in.data_block() );

  /** Get the number of parameters. */
  const unsigned int currentNumberOfPixels =
    this->m_CurrentGridRegion.GetNumberOfPixels();

  /** Create the new vector of output parameters, with the correct size. */
  parameters_out.SetSize(
    this->m_RequiredGridRegion.GetNumberOfPixels() * Dimension );

  /** The input parameters are represented as a coefficient image. */
  ImagePointer coeffs_in = ImageType::New();
  coeffs_in->SetOrigin(  this->m_CurrentGridOrigin );
  coeffs_in->SetSpacing( this->m_CurrentGridSpacing );
  coeffs_in->SetDirection( this->m_CurrentGridDirection );
  coeffs_in->SetRegions( this->m_CurrentGridRegion );

  /** Initialise iterator in the parameters_out. */
  unsigned int i = 0;

  /** Loop over dimension: each direction is upsampled separately. */
  for ( unsigned int j = 0; j < Dimension; j++ )
  {
    /** Fill the coefficient image with parameter data. */
    coeffs_in->GetPixelContainer()
      ->SetImportPointer( inputDataPointer, currentNumberOfPixels );
    inputDataPointer += currentNumberOfPixels;

    /* Set the coeficient image as the input of the upsampler filter.
     * The upsampler samples the deformation field at the locations
     * of the new control points, given the current coefficients
     * (note: it does not just interpolate the coefficient image,
     * which would be wrong). The B-spline coefficients that
     * describe the resulting image are computed by the
     * decomposition filter.
     *
     * This code is copied from the itk-example
     * DeformableRegistration6.cxx .
     */
    typename UpsampleFilterType::Pointer upsampler
      = UpsampleFilterType::New();
    typename CoefficientUpsampleFunctionType::Pointer coeffUpsampleFunction
      = CoefficientUpsampleFunctionType::New();
    typename DecompositionFilterType::Pointer decompositionFilter
      = DecompositionFilterType::New();

    /** Setup the upsampler. */
    upsampler->SetInterpolator( coeffUpsampleFunction );
    upsampler->SetSize( this->m_RequiredGridRegion.GetSize() );
    upsampler->SetOutputStartIndex( this->m_RequiredGridRegion.GetIndex() );
    upsampler->SetOutputSpacing( this->m_RequiredGridSpacing );
    upsampler->SetOutputOrigin( this->m_RequiredGridOrigin );
    upsampler->SetOutputDirection( this->m_RequiredGridDirection );
    upsampler->SetInput( coeffs_in );

    /** Setup the decomposition filter. */
    decompositionFilter->SetSplineOrder( this->m_BSplineOrder );
    decompositionFilter->SetInput( upsampler->GetOutput() );

    /** Do the upsampling. */
    try
    {
      decompositionFilter->UpdateLargestPossibleRegion();
    }
    catch( itk::ExceptionObject & excp )
    {
      /** Add information to the exception. */
      excp.SetLocation( "UpsampleBSplineParametersFilter - UpsampleParameters()" );
      std::string err_str = excp.GetDescription();
      err_str += "\nError occurred while using decompositionFilter.\n";
      excp.SetDescription( err_str );

      /** Pass the exception to an higher level. */
      throw excp;
    }

    /** Get a pointer to the upsampled coefficient image. */
    ImagePointer coeffs_out = decompositionFilter->GetOutput();

    /** Create an iterator over the new coefficient image. */
    IteratorType iterator( coeffs_out, this->m_RequiredGridRegion );
    iterator.GoToBegin();
    while ( !iterator.IsAtEnd() )
    {
      /** Copy the contents of coeffs_out in a ParametersType array. */
      parameters_out[ i ] = iterator.Get();
      ++iterator;
      ++i;
    } // end while coeffs_out iterator loop

  } // end for dimension loop

} // end UpsampleParameters()


/**
 * ******************* DoUpsampling *******************
 */

template< class TArray, class TImage >
bool
UpsampleBSplineParametersFilter<TArray,TImage>
::DoUpsampling( void )
{
  bool ret = ( this->m_CurrentGridOrigin != this->m_RequiredGridOrigin );
  ret |= ( this->m_CurrentGridSpacing != this->m_RequiredGridSpacing );
  ret |= ( this->m_CurrentGridDirection != this->m_RequiredGridDirection );
  ret |= ( this->m_CurrentGridRegion != this->m_RequiredGridRegion );

  return ret;

} // end DoUpsampling()


/**
 * ******************* PrintSelf *******************
 */

template< class TArray, class TImage >
void
UpsampleBSplineParametersFilter<TArray,TImage>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  this->Superclass::PrintSelf( os, indent );

  os << indent << "CurrentGridOrigin: "  << this->m_CurrentGridOrigin << std::endl;
  os << indent << "CurrentGridSpacing: " << this->m_CurrentGridSpacing << std::endl;
  os << indent << "CurrentGridDirection: " << this->m_CurrentGridDirection << std::endl;
  os << indent << "CurrentGridRegion: "  << this->m_CurrentGridRegion << std::endl;

  os << indent << "RequiredGridOrigin: "  << this->m_RequiredGridOrigin << std::endl;
  os << indent << "RequiredGridSpacing: " << this->m_RequiredGridSpacing << std::endl;
  os << indent << "RequiredGridDirection: " << this->m_RequiredGridDirection << std::endl;
  os << indent << "RequiredGridRegion: "  << this->m_RequiredGridRegion << std::endl;

  os << indent << "BSplineOrder: " << this->m_BSplineOrder << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif
