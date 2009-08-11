/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

/** version of original itk file on which code is based: */
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkBSplineDeformableTransform.txx,v $
  Language:  C++
  Date:      $Date: 2008-05-08 23:22:35 $
  Version:   $Revision: 1.41 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAdvancedBSplineDeformableTransform_txx
#define __itkAdvancedBSplineDeformableTransform_txx

#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkContinuousIndex.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkIdentityTransform.h"
#include "vnl/vnl_math.h"

namespace itk
{

// Constructor with default arguments
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::AdvancedBSplineDeformableTransform():Superclass(SpaceDimension,0)
{
  // Instantiate weights functions
  this->m_WeightsFunction = WeightsFunctionType::New();
  this->m_DerivativeWeightsFunction = DerivativeWeightsFunctionType::New();
  this->m_SODerivativeWeightsFunction = SODerivativeWeightsFunctionType::New();;
  this->m_SupportSize = this->m_WeightsFunction->GetSupportSize();

  // Instantiate an identity transform
  typedef IdentityTransform<ScalarType, SpaceDimension> IdentityTransformType;
  typename IdentityTransformType::Pointer id = IdentityTransformType::New();
  this->m_BulkTransform = id;

  // Default grid size is zero
  typename RegionType::SizeType size;
  typename RegionType::IndexType index;
  size.Fill( 0 );
  index.Fill( 0 );
  this->m_GridRegion.SetSize( size );
  this->m_GridRegion.SetIndex( index );

  this->m_GridOrigin.Fill( 0.0 );  // default origin is all zeros
  this->m_GridSpacing.Fill( 1.0 ); // default spacing is all ones
  this->m_GridDirection.SetIdentity(); // default spacing is all ones
  this->m_GridOffsetTable.Fill( 0 );

  this->m_InternalParametersBuffer = ParametersType(0);
  // Make sure the parameters pointer is not NULL after construction.
  this->m_InputParametersPointer = &(this->m_InternalParametersBuffer);

  // Initialize coeffient images
  for ( unsigned int j = 0; j < SpaceDimension; j++ )
    {
    this->m_WrappedImage[j] = ImageType::New();
    this->m_WrappedImage[j]->SetRegions( this->m_GridRegion );
    this->m_WrappedImage[j]->SetOrigin( this->m_GridOrigin.GetDataPointer() );
    this->m_WrappedImage[j]->SetSpacing( this->m_GridSpacing.GetDataPointer() );
    this->m_WrappedImage[j]->SetDirection( this->m_GridDirection );
    this->m_CoefficientImage[j] = NULL;
    }

  // Setup variables for computing interpolation
  this->m_Offset = SplineOrder / 2;
  if ( SplineOrder % 2 ) 
    {
    this->m_SplineOrderOdd = true;
    }
  else
    {
    this->m_SplineOrderOdd = false;
    }
  this->m_ValidRegion = this->m_GridRegion;

  // Initialize Jacobian images
  for ( unsigned int j = 0; j < SpaceDimension; j++ )
    {
    this->m_JacobianImage[j] = ImageType::New();
    this->m_JacobianImage[j]->SetRegions( this->m_GridRegion );
    this->m_JacobianImage[j]->SetOrigin( this->m_GridOrigin.GetDataPointer() );
    this->m_JacobianImage[j]->SetSpacing( this->m_GridSpacing.GetDataPointer() );
    this->m_JacobianImage[j]->SetDirection( this->m_GridDirection );
    }

  /** Fixed Parameters store the following information:
   *     Grid Size
   *     Grid Origin
   *     Grid Spacing
   *     Grid Direction
   *  The size of these is equal to the  NInputDimensions
   */
  this->m_FixedParameters.SetSize ( NDimensions * (NDimensions + 3) );
  this->m_FixedParameters.Fill ( 0.0 );
  for (unsigned int i=0; i<NDimensions; i++)
    {
    this->m_FixedParameters[2*NDimensions+i] = this->m_GridSpacing[i];
    }
  for (unsigned int di=0; di<NDimensions; di++)
    {
    for (unsigned int dj=0; dj<NDimensions; dj++)
      {
      this->m_FixedParameters[3*NDimensions+(di*NDimensions+dj)]
        = this->m_GridDirection[di][dj];
      }
    }

  DirectionType scale;
  for( unsigned int i=0; i<SpaceDimension; i++)
    {
    scale[i][i] = this->m_GridSpacing[i];
    }

  this->m_IndexToPoint = this->m_GridDirection * scale;
  this->m_PointToIndexMatrix = this->m_IndexToPoint.GetInverse();
  
  this->m_LastJacobianIndex = this->m_ValidRegion.GetIndex();
  
}
    

// Destructor
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::~AdvancedBSplineDeformableTransform()
{

}


// Get the number of parameters
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
unsigned int
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::GetNumberOfParameters(void) const
{

  // The number of parameters equal SpaceDimension * number of
  // of pixels in the grid region.
  return ( static_cast<unsigned int>( SpaceDimension ) *
           static_cast<unsigned int>( this->m_GridRegion.GetNumberOfPixels() ) );

}


// Get the number of parameters per dimension
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
unsigned int
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::GetNumberOfParametersPerDimension(void) const
{
  // The number of parameters per dimension equal number of
  // of pixels in the grid region.
  return ( static_cast<unsigned int>( this->m_GridRegion.GetNumberOfPixels() ) );

}


// Set the grid region
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::SetGridRegion( const RegionType& region )
{
  if ( this->m_GridRegion != region )
    {

    this->m_GridRegion = region;

    // set regions for each coefficient and Jacobian image
    for ( unsigned int j = 0; j < SpaceDimension; j++ )
      {
      this->m_WrappedImage[j]->SetRegions( this->m_GridRegion );
      this->m_JacobianImage[j]->SetRegions( this->m_GridRegion );
      }

    // Set the valid region
    // If the grid spans the interval [start, last].
    // The valid interval for evaluation is [start+offset, last-offset]
    // when spline order is even.
    // The valid interval for evaluation is [start+offset, last-offset)
    // when spline order is odd.
    // Where offset = vcl_floor(spline / 2 ).
    // Note that the last pixel is not included in the valid region
    // with odd spline orders.
    typename RegionType::SizeType size = this->m_GridRegion.GetSize();
    typename RegionType::IndexType index = this->m_GridRegion.GetIndex();
    for ( unsigned int j = 0; j < SpaceDimension; j++ )
      {
      index[j] += 
        static_cast< typename RegionType::IndexValueType >( this->m_Offset );
      size[j] -= 
        static_cast< typename RegionType::SizeValueType> ( 2 * this->m_Offset );
      this->m_ValidRegionLast[j] = index[j] +
        static_cast< typename RegionType::IndexValueType >( size[j] ) - 1;
      }
    this->m_ValidRegion.SetSize( size );
    this->m_ValidRegion.SetIndex( index );

    this->UpdateGridOffsetTable();

    //
    // If we are using the default parameters, update their size and set to identity.
    //
    
    // Input parameters point to internal buffer => using default parameters.
    if (this->m_InputParametersPointer == &(this->m_InternalParametersBuffer) )
      {
      // Check if we need to resize the default parameter buffer.
      if ( this->m_InternalParametersBuffer.GetSize() != this->GetNumberOfParameters() )
        {
        this->m_InternalParametersBuffer.SetSize( this->GetNumberOfParameters() );
        // Fill with zeros for identity.
        this->m_InternalParametersBuffer.Fill( 0 );
        }
      }

    this->Modified();
    }
}


//
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::UpdateGridOffsetTable( void )
{
  SizeType gridSize = this->m_GridRegion.GetSize();
  this->m_GridOffsetTable.Fill( 1 );
  for ( unsigned int j = 1; j < SpaceDimension; j++ )
  {
    this->m_GridOffsetTable[ j ]
      = this->m_GridOffsetTable[ j - 1 ] * gridSize[ j - 1 ];
  }

} // end UpdateGridOffsetTable()

// Set the grid spacing
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::SetGridSpacing( const SpacingType& spacing )
{
  if ( this->m_GridSpacing != spacing )
    {
    this->m_GridSpacing = spacing;

    // set spacing for each coefficient and Jacobian image
    for ( unsigned int j = 0; j < SpaceDimension; j++ )
      {
      this->m_WrappedImage[j]->SetSpacing( this->m_GridSpacing.GetDataPointer() );
      this->m_JacobianImage[j]->SetSpacing( this->m_GridSpacing.GetDataPointer() );
      }

    DirectionType scale;
    for( unsigned int i=0; i<SpaceDimension; i++)
      {
      scale[i][i] = this->m_GridSpacing[i];
      }

    this->m_IndexToPoint = this->m_GridDirection * scale;
    this->m_PointToIndexMatrix = this->m_IndexToPoint.GetInverse();

    this->Modified();
    }

}

// Set the grid direction
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::SetGridDirection( const DirectionType & direction )
{
  if ( this->m_GridDirection != direction )
    {
    this->m_GridDirection = direction;

    // set direction for each coefficient and Jacobian image
    for ( unsigned int j = 0; j < SpaceDimension; j++ )
      {
      this->m_WrappedImage[j]->SetDirection( this->m_GridDirection );
      this->m_JacobianImage[j]->SetDirection( this->m_GridDirection );
      }

    DirectionType scale;
    for( unsigned int i=0; i<SpaceDimension; i++)
      {
      scale[i][i] = this->m_GridSpacing[i];
      }

    this->m_IndexToPoint = this->m_GridDirection * scale;
    this->m_PointToIndexMatrix = this->m_IndexToPoint.GetInverse();

    this->Modified();
    }

}


// Set the grid origin
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::SetGridOrigin( const OriginType& origin )
{
  if ( this->m_GridOrigin != origin )
    {
    this->m_GridOrigin = origin;

    // set spacing for each coefficient and jacobianimage
    for ( unsigned int j = 0; j < SpaceDimension; j++ )
      {
      this->m_WrappedImage[j]->SetOrigin( this->m_GridOrigin.GetDataPointer() );
      this->m_JacobianImage[j]->SetOrigin( this->m_GridOrigin.GetDataPointer() );
      }

    this->Modified();
    }

}


// Set the parameters
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::SetIdentity()
{
  if( this->m_InputParametersPointer )
    {
    ParametersType * parameters =
      const_cast<ParametersType *>( this->m_InputParametersPointer );
    parameters->Fill( 0.0 );
    this->Modified();
    }
  else 
    {
    itkExceptionMacro( << "Input parameters for the spline haven't been set ! "
       << "Set them using the SetParameters or SetCoefficientImage method first." );
    }
}

// Set the parameters
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::SetParameters( const ParametersType & parameters )
{

  // check if the number of parameters match the
  // expected number of parameters
  if ( parameters.Size() != this->GetNumberOfParameters() )
    {
    itkExceptionMacro(<<"Mismatched between parameters size "
                      << parameters.size() 
                      << " and region size "
                      << this->m_GridRegion.GetNumberOfPixels() );
    }

  // Clean up buffered parameters
  this->m_InternalParametersBuffer = ParametersType( 0 );

  // Keep a reference to the input parameters
  this->m_InputParametersPointer = &parameters;

  // Wrap flat array as images of coefficients
  this->WrapAsImages();

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();
}

// Set the Fixed Parameters
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::SetFixedParameters( const ParametersType & passedParameters )
{
 
  ParametersType parameters( NDimensions * (3 + NDimensions) );

  // check if the number of parameters match the
  // expected number of parameters
  if ( passedParameters.Size() == NDimensions * 3 )
    {
    parameters.Fill( 0.0 );
    for(unsigned int i=0; i<3 * NDimensions; i++)
      {
      parameters[i] = passedParameters[i];
      }
    for (unsigned int di=0; di<NDimensions; di++)
      {
      parameters[3*NDimensions+(di*NDimensions+di)] = 1;
      }
    }
  else if ( passedParameters.Size() != NDimensions * (3 + NDimensions) )
    {
    itkExceptionMacro(<< "Mismatched between parameters size "
                      << passedParameters.size() 
                      << " and number of fixed parameters "
                      << NDimensions * (3 + NDimensions) );
    }
  else
    {
    for(unsigned int i=0; i<NDimensions * (3 + NDimensions); i++)
      {
      parameters[i] = passedParameters[i];
      }
    }

  /********************************************************* 
    Fixed Parameters store the following information:
        Grid Size
        Grid Origin
        Grid Spacing
        Grid Direction
     The size of these is equal to the  NInputDimensions
  *********************************************************/
  
  /** Set the Grid Parameters */
  SizeType   gridSize;
  for (unsigned int i=0; i<NDimensions; i++)
    {
    gridSize[i] = static_cast<int> (parameters[i]);
    }
  RegionType bsplineRegion;
  bsplineRegion.SetSize( gridSize );
  
  /** Set the Origin Parameters */
  OriginType origin;
  for (unsigned int i=0; i<NDimensions; i++)
    {
    origin[i] = parameters[NDimensions+i];
    }
  
  /** Set the Spacing Parameters */
  SpacingType spacing;
  for (unsigned int i=0; i<NDimensions; i++)
    {
    spacing[i] = parameters[2*NDimensions+i];
    }

  /** Set the Direction Parameters */
  DirectionType direction;
  for (unsigned int di=0; di<NDimensions; di++)
    {
    for (unsigned int dj=0; dj<NDimensions; dj++)
      {
      direction[di][dj] = parameters[3*NDimensions+(di*NDimensions+dj)];
      }
    }
  
  
  this->SetGridSpacing( spacing );
  this->SetGridDirection( direction );
  this->SetGridOrigin( origin );
  this->SetGridRegion( bsplineRegion );
  this->UpdateGridOffsetTable();

  this->Modified();
}


// Wrap flat parameters as images
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::WrapAsImages( void )
{

  /**
   * Wrap flat parameters array into SpaceDimension number of ITK images
   * NOTE: For efficiency, parameters are not copied locally. The parameters
   * are assumed to be maintained by the caller.
   */
  PixelType * dataPointer =
    const_cast<PixelType *>(( this->m_InputParametersPointer->data_block() ));
  unsigned int numberOfPixels = this->m_GridRegion.GetNumberOfPixels();

  for ( unsigned int j = 0; j < SpaceDimension; j++ )
    {
    this->m_WrappedImage[j]->GetPixelContainer()->
      SetImportPointer( dataPointer, numberOfPixels );
    dataPointer += numberOfPixels;
    this->m_CoefficientImage[j] = this->m_WrappedImage[j];
    }

  /**
   * Allocate memory for Jacobian and wrap into SpaceDimension number
   * of ITK images
   */
  this->m_Jacobian.set_size( SpaceDimension, this->GetNumberOfParameters() );
  this->m_Jacobian.Fill( NumericTraits<JacobianPixelType>::Zero );
  this->m_LastJacobianIndex = this->m_ValidRegion.GetIndex();
  JacobianPixelType * jacobianDataPointer = this->m_Jacobian.data_block();

  for ( unsigned int j = 0; j < SpaceDimension; j++ )
    {
    m_JacobianImage[j]->GetPixelContainer()->
      SetImportPointer( jacobianDataPointer, numberOfPixels );
    jacobianDataPointer += this->GetNumberOfParameters() + numberOfPixels;
    }
}


// Set the parameters by value
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::SetParametersByValue( const ParametersType & parameters )
{

  // check if the number of parameters match the
  // expected number of parameters
  if ( parameters.Size() != this->GetNumberOfParameters() )
    {
    itkExceptionMacro(<<"Mismatched between parameters size "
                      << parameters.size() 
                      << " and region size "
                      << this->m_GridRegion.GetNumberOfPixels() );
    }

  // copy it
  this->m_InternalParametersBuffer = parameters;
  this->m_InputParametersPointer = &(this->m_InternalParametersBuffer);

  // wrap flat array as images of coefficients
  this->WrapAsImages();

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();

}

// Get the parameters
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
const 
typename AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::ParametersType &
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::GetParameters( void ) const
{
  /** NOTE: For efficiency, this class does not keep a copy of the parameters - 
   * it just keeps pointer to input parameters. 
   */
  if (NULL == this->m_InputParametersPointer)
    {
    itkExceptionMacro( <<"Cannot GetParameters() because m_InputParametersPointer is NULL. Perhaps SetCoefficientImage() has been called causing the NULL pointer." );
    }

  return (*this->m_InputParametersPointer);
}


// Get the parameters
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
const 
typename AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::ParametersType &
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::GetFixedParameters( void ) const
{
  RegionType resRegion = this->GetGridRegion(  );
  
  for (unsigned int i=0; i<NDimensions; i++)
    {
    this->m_FixedParameters[i] = (resRegion.GetSize())[i];
    }
  for (unsigned int i=0; i<NDimensions; i++)
    {
    this->m_FixedParameters[NDimensions+i] = (this->GetGridOrigin())[i];
    } 
  for (unsigned int i=0; i<NDimensions; i++)
    {
    this->m_FixedParameters[2*NDimensions+i] =  (this->GetGridSpacing())[i];
    }
  for (unsigned int di=0; di<NDimensions; di++)
    {
    for (unsigned int dj=0; dj<NDimensions; dj++)
      {
      this->m_FixedParameters[3*NDimensions+(di*NDimensions+dj)] = (this->GetGridDirection())[di][dj];
      }
    }
  
  return (this->m_FixedParameters);
}


  
// Set the B-Spline coefficients using input images
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void 
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::SetCoefficientImage( ImagePointer images[] )
{
  if ( images[0] )
    {
    this->SetGridRegion( images[0]->GetBufferedRegion() );
    this->SetGridSpacing( images[0]->GetSpacing() );
    this->SetGridDirection( images[0]->GetDirection() );
    this->SetGridOrigin( images[0]->GetOrigin() );
    this->UpdateGridOffsetTable();

    for( unsigned int j = 0; j < SpaceDimension; j++ )
      {
      this->m_CoefficientImage[j] = images[j];
      }

    // Clean up buffered parameters
    this->m_InternalParametersBuffer = ParametersType( 0 );
    this->m_InputParametersPointer  = NULL;

    }

}  

// Print self
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::PrintSelf(std::ostream &os, Indent indent) const
{

  unsigned int j;

  this->Superclass::PrintSelf(os, indent);

  os << indent << "GridRegion: " << this->m_GridRegion << std::endl;
  os << indent << "GridOrigin: " << this->m_GridOrigin << std::endl;
  os << indent << "GridSpacing: " << this->m_GridSpacing << std::endl;
  os << indent << "GridDirection: " << this->m_GridDirection << std::endl;
  os << indent << "GridOffsetTable: " << this->m_GridOffsetTable << std::endl;
  os << indent << "IndexToPoint: " << this->m_IndexToPoint << std::endl;
  os << indent << "PointToIndex: " << this->m_PointToIndexMatrix << std::endl;

  os << indent << "CoefficientImage: [ ";
  for ( j = 0; j < SpaceDimension - 1; j++ )
    {
    os << this->m_CoefficientImage[j].GetPointer() << ", ";
    }
  os << this->m_CoefficientImage[j].GetPointer() << " ]" << std::endl;

  os << indent << "WrappedImage: [ ";
  for ( j = 0; j < SpaceDimension - 1; j++ )
    {
    os << this->m_WrappedImage[j].GetPointer() << ", ";
    }
  os << this->m_WrappedImage[j].GetPointer() << " ]" << std::endl;
 
  os << indent << "InputParametersPointer: " 
     << this->m_InputParametersPointer << std::endl;
  os << indent << "ValidRegion: " << this->m_ValidRegion << std::endl;
  os << indent << "LastJacobianIndex: " << this->m_LastJacobianIndex << std::endl;
  os << indent << "BulkTransform: ";
  os << m_BulkTransform.GetPointer() << std::endl;
  os << indent << "WeightsFunction: ";
  os << this->m_WeightsFunction.GetPointer() << std::endl;

  if ( this->m_BulkTransform )
    {
    os << indent << "BulkTransformType: " 
       << this->m_BulkTransform->GetNameOfClass() << std::endl;
    }
     
}

// Transform a point
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
bool 
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::InsideValidRegion( 
  const ContinuousIndexType& index ) const
{
  bool inside = true;

  if ( !this->m_ValidRegion.IsInside( index ) )
    {
    inside = false;
    }

  if ( inside && this->m_SplineOrderOdd )
    {
    typedef typename ContinuousIndexType::ValueType ValueType;
    for( unsigned int j = 0; j < SpaceDimension; j++ )
      {
      if ( index[j] >= static_cast<ValueType>( this->m_ValidRegionLast[j] ) )
        { 
        inside = false;
        break;
        }
      }
    }

  return inside;

}

// Transform a point
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void 
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::TransformPoint( 
  const InputPointType & point, 
  OutputPointType & outputPoint, 
  WeightsType & weights, 
  ParameterIndexArrayType & indices,
  bool & inside ) const
{
  inside = true;

  /** Take care of the initial transform. */
  InputPointType transformedPoint;
  if ( this->m_BulkTransform )
  {
    transformedPoint = this->m_BulkTransform->TransformPoint( point );
  }
  else
  {
    transformedPoint = point;
  }

  /** Check if the coefficient image has been set. */
  if ( !this->m_CoefficientImage[ 0 ] )
  {
    itkWarningMacro( << "B-spline coefficients have not been set" );
    for ( unsigned int j = 0; j < SpaceDimension; j++ )
    {
      outputPoint[ j ] = transformedPoint[ j ];
    }
    return;
  }

  /***/
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( point, cindex );

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and return the input point
  inside = this->InsideValidRegion( cindex );
  if ( !inside )
  {
    outputPoint = transformedPoint;
    return;
  }

  // Compute interpolation weights
  IndexType supportIndex;
  this->m_WeightsFunction->Evaluate( cindex, weights, supportIndex );

  // For each dimension, correlate coefficient with weights
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  outputPoint.Fill( NumericTraits<ScalarType>::Zero );

  /** Create iterators over the coefficient images. */
  typedef ImageRegionConstIterator<ImageType> IteratorType;
  IteratorType iterator[ SpaceDimension ];
  unsigned long counter = 0;
  const PixelType * basePointer
    = this->m_CoefficientImage[ 0 ]->GetBufferPointer();

  for ( unsigned int j = 0; j < SpaceDimension; j++ )
  {
    iterator[ j ] = IteratorType( this->m_CoefficientImage[ j ], supportRegion );
  }

  /** Loop over the support region. */
  while ( !iterator[ 0 ].IsAtEnd() )
  {
    // populate the indices array
    indices[ counter ] = &(iterator[ 0 ].Value()) - basePointer;

    // multiply weigth with coefficient to compute displacement
    for ( unsigned int j = 0; j < SpaceDimension; j++ )
    {
       outputPoint[ j ] += static_cast<ScalarType>(
         weights[ counter ] * iterator[ j ].Value() );
       ++iterator[ j ];
    }
    ++ counter;

  } // end while

  // The output point is the start point + displacement.
  for ( unsigned int j = 0; j < SpaceDimension; j++ )
  {
    outputPoint[ j ] += transformedPoint[ j ];
  }

}

// Transform a point
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
typename AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::OutputPointType
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::TransformPoint(const InputPointType &point) const 
{  
  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  typename ParameterIndexArrayType::ValueType indicesArray[ numberOfWeights ];
  WeightsType weights( weightsArray, numberOfWeights, false );
  ParameterIndexArrayType indices( indicesArray, numberOfWeights, false );

  OutputPointType outputPoint;
  bool inside;

  this->TransformPoint( point, outputPoint, weights, indices, inside );

  return outputPoint;
}

 
// Compute the Jacobian in one position 
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
const 
typename AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::JacobianType & 
AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
::GetJacobian( const InputPointType & point ) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if( this->m_InputParametersPointer == NULL )
    {
    itkExceptionMacro( <<"Cannot compute Jacobian: parameters not set" );
    }

  // Zero all components of Jacobian
  // NOTE: for efficiency, we only need to zero out the coefficients
  // that got fill last time this method was called.
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( this->m_LastJacobianIndex );

  typedef ImageRegionIterator<JacobianImageType> IteratorType;
  IteratorType iterator[ SpaceDimension ];
  unsigned int j;

  for ( j = 0; j < SpaceDimension; j++ )
    {
    iterator[j] = IteratorType( this->m_JacobianImage[j], supportRegion );
    }

  while ( ! iterator[0].IsAtEnd() )
    {

    // zero out Jacobian elements
    for ( j = 0; j < SpaceDimension; j++ )
      {
      iterator[j].Set( NumericTraits<JacobianPixelType>::Zero );
      }

    for ( j = 0; j < SpaceDimension; j++ )
      {
      ++( iterator[j] );
      }
    }

 
  ContinuousIndexType index;

  this->TransformPointToContinuousGridIndex( point, index );

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and return the input point
  if ( !this->InsideValidRegion( index ) )
    {
    return this->m_Jacobian;
    }

  // Compute interpolation weights
  IndexType supportIndex;
  
  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  WeightsType weights( weightsArray, numberOfWeights, false );  

  this->m_WeightsFunction->Evaluate( index, weights, supportIndex );
  this->m_LastJacobianIndex = supportIndex;

  // For each dimension, copy the weight to the support region
  supportRegion.SetIndex( supportIndex );
  unsigned long counter = 0;

  for ( j = 0; j < SpaceDimension; j++ )
    {
    iterator[j] = IteratorType( this->m_JacobianImage[j], supportRegion );
    }

  while ( !iterator[0].IsAtEnd() )
    {

    // copy weight to Jacobian image
    for ( j = 0; j < SpaceDimension; j++ )
      {
      iterator[j].Set( static_cast<JacobianPixelType>( weights[counter] ) );
      }

    // go to next coefficient in the support region
    ++ counter;
    for ( j = 0; j < SpaceDimension; j++ )
      {
      ++( iterator[j] );
      }
    }


  // Return the results
  return this->m_Jacobian;

}


// Compute the Jacobian in one position 
template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetJacobian( const InputPointType & point, WeightsType& weights, ParameterIndexArrayType& indexes) const
{

  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  const PixelType * basePointer = this->m_CoefficientImage[0]->GetBufferPointer();

  ContinuousIndexType index;

  this->TransformPointToContinuousGridIndex( point, index ); 

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and return the input point
  if ( !this->InsideValidRegion( index ) )
    {
    weights.Fill(0.0);
    indexes.Fill(0);
    return;
    }
  
  // Compute interpolation weights
  IndexType supportIndex;

  this->m_WeightsFunction->Evaluate( index, weights, supportIndex );

  // For each dimension, copy the weight to the support region
  supportRegion.SetIndex( supportIndex );
  unsigned long counter = 0;

  typedef ImageRegionIterator<JacobianImageType> IteratorType;

  IteratorType iterator = IteratorType( this->m_CoefficientImage[0], supportRegion );


  while ( ! iterator.IsAtEnd() )
    {


    indexes[counter] = &(iterator.Value()) - basePointer;

    // go to next coefficient in the support region
    ++ counter;
    ++iterator;
    
    }

}


template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::TransformPointToContinuousGridIndex(
  const InputPointType & point,
  ContinuousIndexType & cindex ) const
{
  Vector<double, SpaceDimension> tvector;

  for ( unsigned int j = 0; j < SpaceDimension; j++ )
  {
    tvector[ j ] = point[ j ] - this->m_GridOrigin[ j ];
  }

  Vector<double, SpaceDimension> cvector
    = this->m_PointToIndexMatrix * tvector;

  for ( unsigned int j = 0; j < SpaceDimension; j++ )
  {
    cindex[ j ] = static_cast< typename ContinuousIndexType::CoordRepType >(
      cvector[ j ] );
  }
}


template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
unsigned int 
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetNumberOfAffectedWeights() const
{
  return this->m_WeightsFunction->GetNumberOfWeights();
}


/**
 * ********************* GetNumberOfNonZeroJacobianIndices ****************************
 */

template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
unsigned long
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetNumberOfNonZeroJacobianIndices( void ) const
{
  return this->m_WeightsFunction->GetNumberOfWeights() * SpaceDimension;

} // end GetNumberOfNonZeroJacobianIndices()


/**
 * ********************* GetJacobian ****************************
 */

template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetJacobian(
  const InputPointType & ipp,
  JacobianType & jacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  /** This implements a sparse version of the Jacobian. */
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if ( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  /** Initialize */
  const unsigned int nnzji = this->GetNumberOfNonZeroJacobianIndices();
  if ( (jacobian.cols() != nnzji) || (jacobian.rows() != SpaceDimension) )
  {
    jacobian.SetSize( SpaceDimension, nnzji );
  }
  jacobian.Fill(0.0);
  
  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and zero Jacobian
  if ( !this->InsideValidRegion( cindex ) )
  { 
    /** Return some dummy */
    nonZeroJacobianIndices.resize( this->GetNumberOfNonZeroJacobianIndices() );
    for (unsigned int i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i )
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }
  
  /** Helper variables. */  
  IndexType supportIndex; 

  /** Compute the number of affected B-spline parameters. */
  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  WeightsType weights( weightsArray, numberOfWeights, false );

  /** Compute the derivative weights. */
  this->m_WeightsFunction->Evaluate( cindex, weights, supportIndex );

  /** Set up support region */
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  /** Put at the right positions */  
  unsigned int counter = 0;
  for ( unsigned int d = 0; d < SpaceDimension; ++d )
  {
    for ( unsigned int mu = 0; mu < numberOfWeights; ++mu )
    {
      jacobian( d, counter ) = weights[mu];
      ++counter;
    }
  } 

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices( nonZeroJacobianIndices, supportRegion );

} // end GetJacobian()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetSpatialJacobian(
  const InputPointType & ipp,
  SpatialJacobianType & sj ) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if ( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  /** Compute the number of affected B-spline parameters. */
  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  WeightsType weights( weightsArray, numberOfWeights, false );

  IndexType supportIndex;
  this->m_DerivativeWeightsFunction->ComputeStartIndex(
    cindex, supportIndex );
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );
  
  /** Compute the spatial Jacobian sj:
     *    dT_{dim} / dx_i = delta_{dim,i} + \sum coefs_{dim} * weights.
     */
  sj.SetIdentity();
  for ( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    /** Set the derivative direction. */
    this->m_DerivativeWeightsFunction->SetDerivativeDirection( i );

    /** Compute the derivative weights. */
    this->m_DerivativeWeightsFunction->Evaluate( cindex, weights, supportIndex );

    /** Compute the spatial Jacobian sj:
     *    dT_{dim} / dx_i = \sum coefs_{dim} * weights.
     */
    for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      /** Create an iterator over the correct part of the coefficient
       * image. Create an iterator over the weights vector.
       */
      ImageRegionConstIterator<ImageType> itCoef(
        this->m_CoefficientImage[ dim ], supportRegion );
      typename WeightsType::const_iterator itWeights = weights.begin();

      /** Compute the sum for this dimension. */
      double sum = 0.0;
      while ( !itCoef.IsAtEnd() )
      {
        sum += itCoef.Value() * (*itWeights);
        ++itWeights;
        ++itCoef;
      }

      /** Update the spatial Jacobian sj. */
      sj( dim, i ) += sum;

    } // end for dim
  } // end for i

} // end GetSpatialJacobian()


/**
 * ********************* GetSpatialHessian ****************************
 */

template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetSpatialHessian(
  const InputPointType & ipp,
  SpatialHessianType & sh ) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if ( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  /** Convert the physical point to a continuous index, which
   * is needed for the evaluate functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  /** Helper variables. */
  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  WeightsType weights( weightsArray, numberOfWeights, false );

  IndexType supportIndex;
  this->m_SODerivativeWeightsFunction->ComputeStartIndex(
    cindex, supportIndex );
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  /** For all derivative directions, compute the spatial Hessian.
   * The derivatives are d^2T / dx_i dx_j.
   * Make use of the fact that the Hessian is symmetrical, so do not compute
   * both i,j and j,i for i != j.
   */
  for ( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    for ( unsigned int j = 0; j <= i; ++j )
    {
      /** Set the derivative direction. */
      this->m_SODerivativeWeightsFunction->SetDerivativeDirections( i, j );

      /** Compute the derivative weights. */
      this->m_SODerivativeWeightsFunction->Evaluate( cindex, weights, supportIndex );
     
      /** Compute d^2T_{dim} / dx_i dx_j = \sum coefs_{dim} * weights. */
      for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
      {
        /** Create an iterator over the correct part of the coefficient image. */
        ImageRegionConstIterator<ImageType> itCoef(
          this->m_CoefficientImage[ dim ], supportRegion );

        /** Compute the sum for this dimension. */
        double sum = 0.0;
        unsigned int mu = 0;
        while ( !itCoef.IsAtEnd() )
        {
          sum += itCoef.Value() * weights[ mu ];
          ++itCoef;
          ++mu;
        }
        
        /** Update the spatial Hessian sh. The Hessian is symmetrical. */
        sh[ dim ][ i ][ j ] = sum;
        sh[ dim ][ j ][ i ] = sum;
      }

    }
  }

} // end GetSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetJacobianOfSpatialJacobian(
  const InputPointType & ipp,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if ( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  jsj.resize( this->GetNumberOfNonZeroJacobianIndices() );

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );
  
  /** Helper variables. */
  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  WeightsType weights( weightsArray, numberOfWeights, false );  
  IndexType supportIndex;
  this->m_DerivativeWeightsFunction->ComputeStartIndex(
    cindex, supportIndex );
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  /** On the stack instead of heap is faster. */
  //double * weightVector = new double[ SpaceDimension * numberOfWeights ];
  double weightVector[ SpaceDimension * numberOfWeights ];

  /** For all derivative directions, compute the derivatives of the
   * spatial Jacobian to the transformation parameters mu:
   * d/dmu of dT / dx_i
   */
  for ( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    /** Set the derivative direction. */
    this->m_DerivativeWeightsFunction->SetDerivativeDirection( i );

    /** Compute the derivative weights. */
    this->m_DerivativeWeightsFunction->Evaluate( cindex, weights, supportIndex );

    /** Remember the weights. */
    memcpy( weightVector + i * numberOfWeights,
      weights.data_block(), numberOfWeights * sizeof( double ) );

  } // end for i

  /** Compute the Jacobian of the spatial Jacobian jsj:
   *    d/dmu dT_{dim} / dx_i = weights.
   */
  SpatialJacobianType * basepointer = &jsj[ 0 ];
  for ( unsigned int mu = 0; mu < numberOfWeights; ++mu )
  {
    for ( unsigned int i = 0; i < SpaceDimension; ++i )
    {
      double tmp = *( weightVector + i * numberOfWeights + mu );
      for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
      {
        (*( basepointer + dim * numberOfWeights + mu ))( dim, i ) = tmp;
      }
    }
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices( nonZeroJacobianIndices, supportRegion );

} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetJacobianOfSpatialJacobian(
  const InputPointType & ipp,
  SpatialJacobianType & sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if ( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  jsj.resize( this->GetNumberOfNonZeroJacobianIndices() );

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  /** Compute the number of affected B-spline parameters. */

  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  WeightsType weights( weightsArray, numberOfWeights, false );

  /** Helper variables. */  
  IndexType supportIndex;
  this->m_DerivativeWeightsFunction->ComputeStartIndex(
    cindex, supportIndex );
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  /** On the stack instead of heap is faster. */
  //double * weightVector = new double[ SpaceDimension * numberOfWeights ];
  double weightVector[ SpaceDimension * numberOfWeights ];

  /** Initialize the spatial Jacobian sj: */
  sj.SetIdentity();

  /** For all derivative directions, compute the derivatives of the
   * spatial Jacobian to the transformation parameters mu:
   * d/dmu of dT / dx_i */
  for ( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    /** Set the derivative direction. */
    this->m_DerivativeWeightsFunction->SetDerivativeDirection( i );

    /** Compute the derivative weights. */
    this->m_DerivativeWeightsFunction->Evaluate( cindex, weights, supportIndex );
    /** \todo: we can realise some speedup here to compute the derivative
     * weights at once for all dimensions */

    /** Remember the weights. */
    memcpy( weightVector + i * numberOfWeights,
      weights.data_block(), numberOfWeights * sizeof( double ) );
    
    /** Compute the spatial Jacobian sj:
     *    dT_{dim} / dx_i = delta_{dim,i} + \sum coefs_{dim} * weights. */
    for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      /** Create an iterator over the correct part of the coefficient
       * image. Create an iterator over the weights vector.
       */
      ImageRegionConstIterator<ImageType> itCoef(
        this->m_CoefficientImage[ dim ], supportRegion );
      typename WeightsType::const_iterator itWeights = weights.begin();

      /** Compute the sum for this dimension. */
      double sum = 0.0;
      while ( !itCoef.IsAtEnd() )
      {
        sum += itCoef.Value() * (*itWeights);
        ++itWeights;
        ++itCoef;
      }

      /** Update the spatial Jacobian sj. */
      sj( dim, i ) += sum;

    } // end for dim
  } // end for i

  /** Compute the Jacobian of the spatial Jacobian jsj:
   *    d/dmu dT_{dim} / dx_i = weights.
   */
  SpatialJacobianType * basepointer = &jsj[ 0 ];
  for ( unsigned int mu = 0; mu < numberOfWeights; ++mu )
  {
    for ( unsigned int i = 0; i < SpaceDimension; ++i )
    {
      double tmp = *( weightVector + i * numberOfWeights + mu );
      for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
      {
        (*( basepointer + dim * numberOfWeights + mu ))( dim, i ) = tmp;
      }
    }
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices( nonZeroJacobianIndices, supportRegion );

} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetJacobianOfSpatialHessian(
  const InputPointType & ipp,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if ( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  jsh.resize( this->GetNumberOfNonZeroJacobianIndices() );

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  /** Compute the number of affected B-spline parameters. */
  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  WeightsType weights( weightsArray, numberOfWeights, false );

  IndexType supportIndex;
  this->m_SODerivativeWeightsFunction->ComputeStartIndex(
    cindex, supportIndex );
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  /** For all derivative directions, compute the derivatives of the
   * spatial Hessian to the transformation parameters mu:
   * d/dmu of d^2T / dx_i dx_j
   * Make use of the fact that the Hessian is symmetrical, so do not compute
   * both i,j and j,i for i != j.
   */
  const unsigned int d = SpaceDimension * ( SpaceDimension + 1 ) / 2;
  FixedArray<WeightsType, d> weightVector;
  unsigned int count = 0;
  for ( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    for ( unsigned int j = 0; j <= i; ++j )
    {
      /** Set the derivative direction. */
      this->m_SODerivativeWeightsFunction->SetDerivativeDirections( i, j );

      /** Compute the derivative weights. */
      this->m_SODerivativeWeightsFunction->Evaluate( cindex, weights, supportIndex );

      /** Remember the weights. */
      weightVector[ count ] = weights;
      ++count;

    } // end for j
  } // end for i

  /** Compute d/dmu d^2T_{dim} / dx_i dx_j = weights. */
  SpatialHessianType * basepointer = &jsh[ 0 ];
  for ( unsigned int mu = 0; mu < numberOfWeights; ++mu )
  {
    SpatialJacobianType matrix;
    unsigned int count = 0;
    for ( unsigned int i = 0; i < SpaceDimension; ++i )
    {
      for ( unsigned int j = 0; j <= i; ++j )
      {
        matrix[ i ][ j ] = weightVector[ count ][ mu ];
        if ( i != j ) matrix[ j ][ i ] = matrix[ i ][ j ];
        ++count;
      }
    }

    for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      (*(basepointer + mu + dim * numberOfWeights))[ dim ] = matrix;
    }
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices( nonZeroJacobianIndices, supportRegion );
  
} // end GetJacobianOfSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::GetJacobianOfSpatialHessian(
  const InputPointType & ipp,
  SpatialHessianType & sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if ( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  jsh.resize( this->GetNumberOfNonZeroJacobianIndices() );

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  ContinuousIndexType cindex;
  this->TransformPointToContinuousGridIndex( ipp, cindex );

  /** Compute the number of affected B-spline parameters. */
  /** Allocate memory on the stack: */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;
  typename WeightsType::ValueType weightsArray[ numberOfWeights ];
  WeightsType weights( weightsArray, numberOfWeights, false );

  IndexType supportIndex;
  this->m_SODerivativeWeightsFunction->ComputeStartIndex(
    cindex, supportIndex );
  RegionType supportRegion;
  supportRegion.SetSize( this->m_SupportSize );
  supportRegion.SetIndex( supportIndex );

  /** On the stack instead of heap is faster. */
  const unsigned int d = SpaceDimension * ( SpaceDimension + 1 ) / 2;
  //double * weightVector = new double[ SpaceDimension * numberOfWeights ];
  double weightVector[ d * numberOfWeights ];

  /** For all derivative directions, compute the derivatives of the
   * spatial Hessian to the transformation parameters mu:
   * d/dmu of d^2T / dx_i dx_j
   * Make use of the fact that the Hessian is symmetrical, so do not compute
   * both i,j and j,i for i != j.
   */
  unsigned int count = 0;
  for ( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    for ( unsigned int j = 0; j <= i; ++j )
    {
      /** Set the derivative direction. */
      this->m_SODerivativeWeightsFunction->SetDerivativeDirections( i, j );

      /** Compute the derivative weights. */
      this->m_SODerivativeWeightsFunction->Evaluate( cindex, weights, supportIndex );

      /** Remember the weights. */
      memcpy( weightVector + count * numberOfWeights,
        weights.data_block(), numberOfWeights * sizeof( double ) );
      count++;

      /** Compute the spatial Hessian sh:
       *    d^2T_{dim} / dx_i dx_j = \sum coefs_{dim} * weights.
       */
      for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
      {
        /** Create an iterator over the correct part of the coefficient
         * image. Create an iterator over the weights vector.
         */
        ImageRegionConstIterator<ImageType> itCoef(
          this->m_CoefficientImage[ dim ], supportRegion );
        typename WeightsType::const_iterator itWeights = weights.begin();

        /** Compute the sum for this dimension. */
        double sum = 0.0;
        while ( !itCoef.IsAtEnd() )
        {
          sum += itCoef.Value() * (*itWeights);
          ++itWeights;
          ++itCoef;
        }
        
        /** Update the spatial Hessian sh. The Hessian is symmetrical. */
        sh[ dim ]( i, j ) = sum;
        sh[ dim ]( j, i ) = sum;
      }

    } // end for j
  } // end for i

  /** Compute the Jacobian of the spatial Hessian jsh:
   *    d/dmu d^2T_{dim} / dx_i dx_j = weights.
   */
  SpatialHessianType * basepointer = &jsh[ 0 ];
  for ( unsigned int mu = 0; mu < numberOfWeights; ++mu )
  {
    SpatialJacobianType matrix;
    unsigned int count = 0;
    for ( unsigned int i = 0; i < SpaceDimension; ++i )
    {
      for ( unsigned int j = 0; j <= i; ++j )
      {
        double tmp = *( weightVector + count * numberOfWeights + mu );
        matrix[ i ][ j ] = tmp;
        if ( i != j ) matrix[ j ][ i ] = tmp;
        ++count;
      }
    }

    for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      (*(basepointer + mu + dim * numberOfWeights))[ dim ] = matrix;
    }
  }

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices( nonZeroJacobianIndices, supportRegion );

} // end GetJacobianOfSpatialHessian()


/**
 * ********************* ComputeNonZeroJacobianIndices ****************************
 */

template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
void
AdvancedBSplineDeformableTransform<TScalarType, NDimensions,VSplineOrder>
::ComputeNonZeroJacobianIndices(
  NonZeroJacobianIndicesType & nonZeroJacobianIndices,
  const RegionType & supportRegion ) const
{
  nonZeroJacobianIndices.resize( this->GetNumberOfNonZeroJacobianIndices() );

  /** Create an iterator over the coefficient image. */
  ImageRegionConstIterator< ImageType >
    it( this->m_CoefficientImage[ 0 ], supportRegion );

  /** Initialize some helper variables. */
  const unsigned long numberOfWeights = WeightsFunctionType::NumberOfWeights;  
  const unsigned long parametersPerDim
    = this->GetNumberOfParametersPerDimension();
  //IndexType ind;
  unsigned long mu = 0;

  /** For all control points in the support region, set which of the
   * indices in the parameter array are non-zero.
   */
  const PixelType * basePointer = this->m_CoefficientImage[0]->GetBufferPointer();
  
  while ( !it.IsAtEnd() )
  {
    /** Get the current index. */
    //ind = it.GetIndex();

    /** Translate the index into a parameter number for the x-direction. */
    //unsigned long parameterNumber = 0;
    //for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    //{
    // parameterNumber += ind[ dim ] * this->m_GridOffsetTable[ dim ];
    //}
    unsigned long parameterNumber = &(it.Value()) - basePointer;

    /** Update the nonZeroJacobianIndices for all directions. */
    for ( unsigned int dim = 0; dim < SpaceDimension; ++dim )
    {
      nonZeroJacobianIndices[ mu + dim * numberOfWeights ]
        = parameterNumber + dim * parametersPerDim;
    }

    /** Increase the iterators. */
    ++it;
    ++mu;

  } // end while

} // end ComputeNonZeroJacobianIndices()


} // namespace

#endif
