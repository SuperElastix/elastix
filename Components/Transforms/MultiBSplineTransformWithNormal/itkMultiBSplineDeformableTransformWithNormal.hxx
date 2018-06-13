/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkMultiBSplineDeformableTransformWithNormal_hxx
#define __itkMultiBSplineDeformableTransformWithNormal_hxx

#include "itkMultiBSplineDeformableTransformWithNormal.h"
#include "itkStatisticsImageFilter.h"
#include "itkApproximateSignedDistanceMapImageFilter.h"
#include "itkGradientImageFilter.h"
#include "itkVectorCastImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkConstantPadImageFilter.h"

namespace itk
{

// Constructor with default arguments
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::MultiBSplineDeformableTransformWithNormal() : Superclass( SpaceDimension )
{
  // By default this class handle a unique Transform
  this->m_NbLabels           = 0;
  this->m_Labels             = 0;
  this->m_LabelsInterpolator = 0;
  this->m_Trans.resize( 1 );
  // keep transform 0 to store parameters that are not kept here (GridSize, ...)
  this->m_Trans[ 0 ] = TransformType::New();
  this->m_Para.resize( 0 );
  this->m_LastJacobian = -1;
  this->m_LocalBases   = ImageBaseType::New();

  this->m_InternalParametersBuffer = ParametersType( 0 );
  // Make sure the parameters pointer is not NULL after construction.
  this->m_InputParametersPointer = &( this->m_InternalParametersBuffer );
}


// Destructor
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::~MultiBSplineDeformableTransformWithNormal()
{}

// Get the number of parameters
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
typename MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >::NumberOfParametersType
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::GetNumberOfParameters( void ) const
{
  if( m_NbLabels > 0 )
  {
    return ( 1 + ( SpaceDimension - 1 ) * m_NbLabels ) * m_Trans[ 0 ]->GetNumberOfParametersPerDimension();
  }
  else
  {
    return 0;
  }
}


// Get the number of parameters per dimension
// FIXME :  Do we need to declare this function ?
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
typename MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >::NumberOfParametersType
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::GetNumberOfParametersPerDimension( void ) const
{
  // FIXME : Depends on which dimension we are speaking here. should check it
  if( m_NbLabels > 0 )
  {
    return m_Trans[ 0 ]->GetNumberOfParametersPerDimension();
  }
  else
  {
    return 0;
  }
}


#define LOOP_ON_LABELS( FUNC, ARGS )          \
  for( unsigned i = 0; i <= m_NbLabels; ++i ) \
  { \
    m_Trans[ i ]->FUNC( ARGS ); \
  }

#define SET_ALL_LABELS( FUNC, TYPE ) \
  template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder > \
  void \
  MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder > \
  ::Set##FUNC( const TYPE _arg ) \
  { \
    if( _arg != this->Get##FUNC() ) \
    { \
      LOOP_ON_LABELS( Set##FUNC, _arg ); \
      this->Modified(); \
    } \
  }

#define GET_FIRST_LABEL( FUNC, TYPE ) \
  template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder > \
  typename MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >::TYPE \
  MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder > \
  ::Get##FUNC() const \
  { \
    return m_Trans[ 0 ]->Get##FUNC(); \
  }

// Set the grid region
SET_ALL_LABELS( GridRegion, RegionType & );

// Get the grid region
GET_FIRST_LABEL( GridRegion, RegionType );

// Set the grid spacing
SET_ALL_LABELS( GridSpacing, SpacingType & );

// Get the grid spacing
GET_FIRST_LABEL( GridSpacing, SpacingType );

// Set the grid direction
SET_ALL_LABELS( GridDirection, DirectionType & );

// Get the grid direction
GET_FIRST_LABEL( GridDirection, DirectionType );

// Set the grid origin
SET_ALL_LABELS( GridOrigin, OriginType & );

// Get the grid origin
GET_FIRST_LABEL( GridOrigin, OriginType );

#undef SET_ALL_LABELS
#undef GET_FIRST_LABEL

template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::SetLabels( ImageLabelType * labels )
{
  typedef StatisticsImageFilter< ImageLabelType > StatisticsType;
  if( labels != this->m_Labels )
  {
    // Save current settings
    this->m_Labels = labels;
    ParametersType para = this->GetFixedParameters();
    typename StatisticsType::Pointer stat = StatisticsType::New();
    stat->SetInput( this->m_Labels );
    stat->Update();
    this->m_NbLabels = stat->GetMaximum() + 1;
    this->m_Trans.resize( this->m_NbLabels + 1 );
    this->m_Para.resize( this->m_NbLabels + 1 );
    for( unsigned i = 0; i <= this->m_NbLabels; ++i )
    {
      this->m_Trans[ i ] = TransformType::New();
    }
    this->m_LabelsInterpolator = ImageLabelInterpolator::New();
    this->m_LabelsInterpolator->SetInputImage( this->m_Labels );
    // Restore settings
    this->SetFixedParameters( para );
  }
}


template< class TScalarType, unsigned int NDimensions >
struct UpdateLocalBases_impl
{
  typedef itk::Vector< TScalarType, NDimensions > VectorType;
  typedef itk::Vector< VectorType, NDimensions >  BaseType;
  typedef itk::Image< VectorType, NDimensions >   ImageVectorType;
  typedef typename ImageVectorType::Pointer       ImageVectorPointer;
  typedef itk::Image< BaseType, NDimensions >     ImageBaseType;
  typedef typename ImageBaseType::Pointer         ImageBasePointer;

  static void Do( ImageBaseType *, ImageVectorType * )
  {
    itkGenericExceptionMacro( << "MultiBSplineDeformableTransformWithNormal only works with 3D image for the moment" );
  }


};

template< class TScalarType >
struct UpdateLocalBases_impl< TScalarType, 2 >
{
  static const unsigned NDimensions = 2;
  typedef itk::Vector< TScalarType, NDimensions > VectorType;
  typedef itk::Vector< VectorType, NDimensions >  BaseType;
  typedef itk::Image< VectorType, NDimensions >   ImageVectorType;
  typedef typename ImageVectorType::Pointer       ImageVectorPointer;
  typedef itk::Image< BaseType, NDimensions >     ImageBaseType;
  typedef typename ImageBaseType::Pointer         ImageBasePointer;

  static void Do( ImageBaseType * bases, ImageVectorType * normals )
  {
    const TScalarType base_x[] = { 1, 0 };
    const TScalarType base_y[] = { 0, 1 };

    typedef itk::NearestNeighborInterpolateImageFunction< ImageVectorType, TScalarType > ImageVectorInterpolator;
    typedef typename ImageVectorInterpolator::Pointer                                    ImageVectorInterpolatorPointer;
    ImageVectorInterpolatorPointer vinterp = ImageVectorInterpolator::New();
    vinterp->SetInputImage( normals );

    typedef ImageRegionIterator< ImageBaseType > IteratorType;
    IteratorType it( bases, bases->GetLargestPossibleRegion() );
    for( it.GoToBegin(); !it.IsAtEnd(); ++it )
    {
      BaseType b;
      typename ImageBaseType::PointType p;
      bases->TransformIndexToPhysicalPoint( it.GetIndex(), p );
      typename ImageBaseType::IndexType idx;
      vinterp->ConvertPointToNearestIndex( p, idx );
      if( !vinterp->IsInsideBuffer( idx ) )
      {
        // far from an interface keep (x,y,z) base
        b[ 0 ] = VectorType( base_x );
        b[ 1 ] = VectorType( base_y );
        it.Set( b );
        continue;
      }

      VectorType n = vinterp->EvaluateAtIndex( idx );
      n.Normalize();
      if( n.GetNorm() < 0.1 )
      {
        std::cout << "Should never append" << std::endl;
        // far from an interface keep (x,y,z) base
        b[ 0 ] = VectorType( base_x );
        b[ 1 ] = VectorType( base_y );
        it.Set( b );
        continue;
      }

      b[ 0 ]      = n;
      b[ 1 ][ 0 ] = n[ 1 ];
      b[ 1 ][ 1 ] = -n[ 0 ];
      it.Set( b );
    }
  }


};

template< class TScalarType >
struct UpdateLocalBases_impl< TScalarType, 3 >
{
  static const unsigned NDimensions = 3;
  typedef itk::Vector< TScalarType, NDimensions > VectorType;
  typedef itk::Vector< VectorType, NDimensions >  BaseType;
  typedef itk::Image< VectorType, NDimensions >   ImageVectorType;
  typedef typename ImageVectorType::Pointer       ImageVectorPointer;
  typedef itk::Image< BaseType, NDimensions >     ImageBaseType;
  typedef typename ImageBaseType::Pointer         ImageBasePointer;

  static void Do( ImageBaseType * bases, ImageVectorType * normals )
  {
    const TScalarType base_x[] = { 1, 0, 0 };
    const TScalarType base_y[] = { 0, 1, 0 };
    const TScalarType base_z[] = { 0, 0, 1 };

    typedef itk::NearestNeighborInterpolateImageFunction< ImageVectorType, TScalarType > ImageVectorInterpolator;
    typedef typename ImageVectorInterpolator::Pointer                                    ImageVectorInterpolatorPointer;
    ImageVectorInterpolatorPointer vinterp = ImageVectorInterpolator::New();
    vinterp->SetInputImage( normals );

    typedef ImageRegionIterator< ImageBaseType > IteratorType;
    IteratorType it( bases, bases->GetLargestPossibleRegion() );
    for( it.GoToBegin(); !it.IsAtEnd(); ++it )
    {
      BaseType b;
      typename ImageBaseType::PointType p;
      bases->TransformIndexToPhysicalPoint( it.GetIndex(), p );
      typename ImageBaseType::IndexType idx;
      vinterp->ConvertPointToNearestIndex( p, idx );
      if( !vinterp->IsInsideBuffer( idx ) )
      {
        // far from an interface keep (x,y,z) base
        b[ 0 ] = VectorType( base_x );
        b[ 1 ] = VectorType( base_y );
        b[ 2 ] = VectorType( base_z );
        it.Set( b );
        continue;
      }

      VectorType n = vinterp->EvaluateAtIndex( idx );
      n.Normalize();
      if( n.GetNorm() < 0.1 )
      {
        std::cout << "Should never append" << std::endl;
        // far from an interface keep (x,y,z) base
        b[ 0 ] = VectorType( base_x );
        b[ 1 ] = VectorType( base_y );
        b[ 2 ] = VectorType( base_z );
        it.Set( b );
        continue;
      }

      b[ 0 ] = n;

      // find the must non colinear to vector wrt n
      VectorType tmp;
      if( std::abs( n[ 0 ] ) < std::abs( n[ 1 ] ) )
      {
        if( std::abs( n[ 0 ] ) < std::abs( n[ 2 ] ) )
        {
          tmp = base_x;
        }
        else
        {
          tmp = base_z;
        }
      }
      else
      {
        if( std::abs( n[ 1 ] ) < std::abs( n[ 2 ] ) )
        {
          tmp = base_y;
        }
        else
        {
          tmp = base_z;
        }
      }

      // find u and v in order to form a local orthonormal base with n
      tmp = CrossProduct( n, tmp );
      tmp.Normalize();
      b[ 1 ] = tmp;
      tmp    = CrossProduct( n, tmp );
      tmp.Normalize();
      b[ 2 ] = tmp;
      it.Set( b );
    }
  }


};

template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::UpdateLocalBases()
{
  typedef itk::Image< double, itkGetStaticConstMacro( SpaceDimension ) >                          ImageDoubleType;
  typedef itk::ConstantPadImageFilter< ImageLabelType, ImageLabelType >                           PadFilterType;
  typedef itk::ApproximateSignedDistanceMapImageFilter< ImageLabelType, ImageDoubleType >         DistFilterType;
  typedef itk::SmoothingRecursiveGaussianImageFilter< ImageDoubleType, ImageDoubleType >          SmoothFilterType;
  typedef itk::GradientImageFilter< ImageDoubleType, double, double >                             GradFilterType;
  typedef itk::VectorCastImageFilter< typename GradFilterType::OutputImageType, ImageVectorType > CastVectorType;
  typedef itk::BinaryThresholdImageFilter< ImageLabelType, ImageLabelType >                       LabelExtractorType;
  typedef itk::AddImageFilter< ImageVectorType, ImageVectorType, ImageVectorType >                AddVectorImageType;
  typedef itk::MaskImageFilter< ImageVectorType, ImageLabelType, ImageVectorType >                MaskVectorImageType;
  typedef typename ImageLabelType::PointType                                                      PointType;
  typedef typename ImageLabelType::RegionType                                                     RegionType;
  typedef typename ImageLabelType::SpacingType                                                    SpacingType;

  PointType transOrig = GetGridOrigin();
  PointType transEnd;
  for( unsigned i = 0; i < NDimensions; ++i )
  {
    transEnd[ i ] = transOrig[ i ] + ( GetGridRegion().GetSize()[ i ] - GetGridRegion().GetIndex()[ i ] ) * GetGridSpacing()[ i ];
  }

  PointType   labelOrig = this->m_Labels->GetOrigin();
  RegionType  labelReg  = this->m_Labels->GetLargestPossibleRegion();
  SpacingType labelSpac = this->m_Labels->GetSpacing();
  PointType   labelEnd;
  for( unsigned i = 0; i < NDimensions; ++i )
  {
    labelEnd[ i ] = labelOrig[ i ] + ( labelReg.GetSize()[ i ] - labelReg.GetIndex()[ i ] ) * labelSpac[ i ];
  }

  typename ImageLabelType::SizeType lowerExtend;
  for( unsigned i = 0; i < NDimensions; ++i )
  {
    lowerExtend[ i ] = std::ceil( ( labelOrig[ i ] - transOrig[ i ] ) / labelSpac[ i ] );
  }

  typename ImageLabelType::SizeType upperExtend;
  for( unsigned i = 0; i < NDimensions; ++i )
  {
    upperExtend[ i ] = std::ceil( ( transEnd[ i ] - labelEnd[ i ] ) ) / labelSpac[ i ];
  }

  typename PadFilterType::Pointer padFilter = PadFilterType::New();
  padFilter->SetInput( this->m_Labels );
  padFilter->SetPadLowerBound( lowerExtend );
  padFilter->SetPadUpperBound( upperExtend );
  padFilter->SetConstant( 0 );

  for( int l = 0; l < this->m_NbLabels; ++l )
  {
    typename LabelExtractorType::Pointer labelExtractor = LabelExtractorType::New();
    labelExtractor->SetInput( padFilter->GetOutput() );
    labelExtractor->SetLowerThreshold( l );
    labelExtractor->SetUpperThreshold( l );
    labelExtractor->SetInsideValue( 1 );
    labelExtractor->SetOutsideValue( 0 );

    typename DistFilterType::Pointer distFilter = DistFilterType::New();
    distFilter->SetInsideValue( 1 );
    distFilter->SetOutsideValue( 0 );
    distFilter->SetInput( labelExtractor->GetOutput() );

    typename SmoothFilterType::Pointer smoothFilter = SmoothFilterType::New();
    smoothFilter->SetInput( distFilter->GetOutput() );
    smoothFilter->SetSigma( 4. );

    typename GradFilterType::Pointer gradFilter = GradFilterType::New();
    gradFilter->SetInput( smoothFilter->GetOutput() );

    typename CastVectorType::Pointer castFilter = CastVectorType::New();
    castFilter->SetInput( gradFilter->GetOutput() );

    typename MaskVectorImageType::Pointer maskFilter = MaskVectorImageType::New();
    maskFilter->SetInput( castFilter->GetOutput() );
    maskFilter->SetMaskImage( labelExtractor->GetOutput() );
    maskFilter->SetOutsideValue( itk::NumericTraits< VectorType >::ZeroValue() );
    maskFilter->Update();

    if( l == 0 )
    {
      this->m_LabelsNormals = maskFilter->GetOutput();
    }
    else
    {
      typename AddVectorImageType::Pointer addFilter = AddVectorImageType::New();
      addFilter->SetInput1( this->m_LabelsNormals );
      addFilter->SetInput2( maskFilter->GetOutput() );
      addFilter->Update();
      this->m_LabelsNormals = addFilter->GetOutput();
    }
  }

  m_LocalBases = ImageBaseType::New();
  m_LocalBases->SetRegions( GetGridRegion() );
  m_LocalBases->SetSpacing( GetGridSpacing() );
  m_LocalBases->SetOrigin( GetGridOrigin() );
  m_LocalBases->SetDirection( GetGridDirection() );
  m_LocalBases->Allocate();
  UpdateLocalBases_impl< TScalarType, NDimensions >::Do( this->m_LocalBases, this->m_LabelsNormals );
}


// Set the parameters
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::SetIdentity()
{
  LOOP_ON_LABELS( SetIdentity, );
  if( this->m_InputParametersPointer )
  {
    ParametersType * parameters
      = const_cast< ParametersType * >( this->m_InputParametersPointer );
    parameters->Fill( 0.0 );
    this->Modified();
  }
  else
  {
    itkExceptionMacro( << "Input parameters for the spline haven't been set ! "
                       << "Set them using the SetParameters or SetCoefficientImage method first." );
  }
}


template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::DispatchParameters( const ParametersType & parameters )
{
  for( unsigned i = 0; i <= m_NbLabels; ++i )
  {
    m_Para[ i ].SetSize( m_Trans[ i ]->GetNumberOfParameters() );
  }

  typedef typename ImageBaseType::PixelContainer BaseContainer;
  const BaseContainer & bases                  = *m_LocalBases->GetPixelContainer();
  unsigned              ParametersPerDimension = m_Trans[ 0 ]->GetNumberOfParametersPerDimension();
  for( unsigned i = 0; i < ParametersPerDimension; ++i )
  {
    VectorType tmp = bases[ i ][ 0 ] * parameters.GetElement( i );
    for( unsigned d = 0; d < itkGetStaticConstMacro( SpaceDimension ); ++d )
    {
      m_Para[ 0 ].SetElement( i + d * ParametersPerDimension, tmp[ d ] );
    }

    for( unsigned l = 1; l <= m_NbLabels; ++l )
    {
      for( unsigned d = 0; d < itkGetStaticConstMacro( SpaceDimension ); ++d )
      {
        tmp[ d ] = 0;
      }

      for( unsigned d = 1; d < itkGetStaticConstMacro( SpaceDimension ); ++d )
      {
        tmp += bases[ i ][ d ] * parameters.GetElement( i + ( ( itkGetStaticConstMacro( SpaceDimension ) - 1 )
          * ( l - 1 ) + d )
          * ParametersPerDimension );
      }

      for( unsigned d = 0; d < itkGetStaticConstMacro( SpaceDimension ); ++d )
      {
        m_Para[ l ].SetElement( i + d * ParametersPerDimension, tmp[ d ] );
      }
    }
  }
  for( unsigned i = 0; i <= m_NbLabels; ++i )
  {
    m_Trans[ i ]->SetParameters( m_Para[ i ] );
  }
}


// Set the parameters
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::SetParameters( const ParametersType & parameters )
{
  // check if the number of parameters match the
  // expected number of parameters
  if( parameters.Size() != this->GetNumberOfParameters() )
  {
    itkExceptionMacro( << "Mismatched between parameters size "
                       << parameters.size()
                       << " and region size "
                       << this->GetNumberOfParameters() );
  }

  // Clean up buffered parameters
  this->m_InternalParametersBuffer = ParametersType( 0 );

  // Keep a reference to the input parameters
  this->m_InputParametersPointer = &parameters;

  DispatchParameters( parameters );

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();
}


// Set the Fixed Parameters
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::SetFixedParameters( const ParametersType & passedParameters )
{
  LOOP_ON_LABELS( SetFixedParameters, passedParameters );
  this->Modified();
}


// Set the parameters by value
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::SetParametersByValue( const ParametersType & parameters )
{
  // check if the number of parameters match the
  // expected number of parameters
  if( parameters.Size() != this->GetNumberOfParameters() )
  {
    itkExceptionMacro( << "Mismatched between parameters size "
                       << parameters.size()
                       << " and region size "
                       << this->GetNumberOfParameters() );
  }

  // copy it
  this->m_InternalParametersBuffer = parameters;
  this->m_InputParametersPointer   = &( this->m_InternalParametersBuffer );

  DispatchParameters( parameters );

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();

}


// Get the parameters
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
const
typename MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::ParametersType
& MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::GetParameters( void ) const
{
  /** NOTE: For efficiency, this class does not keep a copy of the parameters -
   * it just keeps pointer to input parameters.
   */
  if( NULL == this->m_InputParametersPointer )
  {
    itkExceptionMacro(
        << "Cannot GetParameters() because m_InputParametersPointer is NULL. Perhaps SetCoefficientImages() has been called causing the NULL pointer." );
  }

  return ( *this->m_InputParametersPointer );
}

// Get the parameters
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
const
typename MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::ParametersType
& MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::GetFixedParameters( void ) const
{
  return ( m_Trans[ 0 ]->GetFixedParameters() );
}

// Print self
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  this->Superclass::PrintSelf( os, indent );

  os << indent << "NbLabels : " << m_NbLabels << std::endl;
  itk::Indent ind = indent.GetNextIndent();

  os << indent << "Normal " << std::endl;
  m_Trans[ 0 ]->Print( os, ind );
  for( unsigned i = 1; i <= m_NbLabels; ++i )
  {
    os << indent << "Label " << i << std::endl;
    m_Trans[ i ]->Print( os, ind );
  }
}


#undef LOOP_ON_LABELS

template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
inline void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::PointToLabel( const InputPointType & p, int & l ) const
{
  l = 0;
  assert( this->m_Labels );
  typename ImageLabelInterpolator::IndexType idx;
  this->m_LabelsInterpolator->ConvertPointToNearestIndex( p, idx );
  if( this->m_LabelsInterpolator->IsInsideBuffer( idx ) )
  {
    l = static_cast< int >( this->m_LabelsInterpolator->EvaluateAtIndex( idx ) ) + 1;
  }
}


template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
typename MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >::OutputPointType
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::TransformPoint( const InputPointType & point ) const
{
  int lidx = 0;
  this->PointToLabel( point, lidx );
  if( lidx == 0 )
  {
    return point;
  }

  OutputPointType res = m_Trans[ 0 ]->TransformPoint( point ) + ( m_Trans[ lidx ]->TransformPoint( point ) - point );
  return res;
}


//template<class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
//const typename MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>::JacobianType&
//MultiBSplineDeformableTransformWithNormal<TScalarType, NDimensions, VSplineOrder>
//::GetJacobian( const InputPointType & point ) const
//{
//  this->m_Jacobian.set_size(SpaceDimension, this->GetNumberOfParameters());
//  this->m_Jacobian.Fill(0.0);
//  JacobianType jacobian;
//  NonZeroJacobianIndicesType nonZeroJacobianIndices;
//  this->GetJacobian(point, jacobian, nonZeroJacobianIndices);
//  for (unsigned i = 0; i < nonZeroJacobianIndices.size(); ++i)
//    for (unsigned j = 0; j < SpaceDimension; ++j)
//    this->m_Jacobian[j][nonZeroJacobianIndices[i]] = jacobian[j][i];
//  return this->m_Jacobian;
//}

/**
 * ********************* GetJacobian ****************************
 */

template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::GetJacobian(
  const InputPointType & ipp,
  JacobianType & jacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  if( this->GetNumberOfParameters() == 0 )
  {
    jacobian.SetSize( SpaceDimension, 0 );
    nonZeroJacobianIndices.resize( 0 );
    return;
  }

  // Initialize
  const unsigned int nnzji = this->GetNumberOfNonZeroJacobianIndices();
  if( ( jacobian.cols() != nnzji ) || ( jacobian.rows() != SpaceDimension ) )
  {
    jacobian.SetSize( SpaceDimension, nnzji );
  }
  jacobian.Fill( 0.0 );

  // This implements a sparse version of the Jacobian.
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  int lidx = 0;
  PointToLabel( ipp, lidx );

  if( lidx == 0 )
  {
    // Return some dummy
    nonZeroJacobianIndices.resize( this->GetNumberOfNonZeroJacobianIndices() );
    for( unsigned int i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i )
    {
      nonZeroJacobianIndices[ i ] = i;
    }
    return;
  }

  JacobianType njac, ljac;
  njac.SetSize( SpaceDimension, nnzji );
  ljac.SetSize( SpaceDimension, nnzji );

  // nzji should be the same so keep only one
  m_Trans[ 0 ]->GetJacobian( ipp, njac, nonZeroJacobianIndices );
  m_Trans[ lidx ]->GetJacobian( ipp, ljac, nonZeroJacobianIndices );

  // Convert the physical point to a continuous index, which
  // is needed for the 'Evaluate()' functions below.
  typename TransformType::ContinuousIndexType cindex;
  m_Trans[ lidx ]->TransformPointToContinuousGridIndex( ipp, cindex );

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and zero Jacobian
  if( !m_Trans[ lidx ]->InsideValidRegion( cindex ) )
  {
    // Return some dummy
    nonZeroJacobianIndices.resize( m_Trans[ lidx ]->GetNumberOfNonZeroJacobianIndices() );
    for( unsigned int i = 0; i < m_Trans[ lidx ]->GetNumberOfNonZeroJacobianIndices(); ++i )
    {
      nonZeroJacobianIndices[ i ] = i;
    }
    return;
  }

  typedef typename ImageBaseType::PixelContainer BaseContainer;
  const BaseContainer & bases = *m_LocalBases->GetPixelContainer();

  const unsigned nweights = this->GetNumberOfWeights();
  for( unsigned i = 0; i < nweights; ++i )
  {
    VectorType tmp = bases[ nonZeroJacobianIndices[ i ] ][ 0 ];
    for( unsigned j = 0; j < SpaceDimension; ++j )
    {
      jacobian[ j ][ i ] = tmp[ j ] * njac[ j ][ i + j * nweights ];
    }

    for( unsigned d = 1; d < SpaceDimension; ++d )
    {
      tmp = bases[ nonZeroJacobianIndices[ i ] ][ d ];
      for( unsigned j = 0; j < SpaceDimension; ++j )
      {
        jacobian[ j ][ i + d * nweights ] = tmp[ j ] * ljac[ j ][ i + j * nweights ];
      }
    }
  }

  // move non zero indices to match label positions
  if( lidx > 1 )
  {
    unsigned to_add = ( lidx - 1 ) * m_Trans[ 0 ]->GetNumberOfParametersPerDimension() * ( SpaceDimension - 1 );
    for( unsigned i = 0; i < nweights; ++i )
    {
      for( unsigned d = 1; d < SpaceDimension; ++d )
      {
        nonZeroJacobianIndices[ d * nweights + i ] += to_add;
      }
    }
  }
} // end GetJacobian()


template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::GetSpatialJacobian( const InputPointType & ipp,
  SpatialJacobianType & sj ) const
{
  if( this->GetNumberOfParameters() == 0 )
  {
    sj.SetIdentity();
    return;
  }

  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  int lidx = 0;
  PointToLabel( ipp, lidx );
  if( lidx == 0 )
  {
    sj.SetIdentity();
    return;
  }
  SpatialJacobianType nsj;
  m_Trans[ 0 ]->GetSpatialJacobian( ipp, nsj );
  m_Trans[ lidx ]->GetSpatialJacobian( ipp, sj );
  sj += nsj;
}


template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::GetSpatialHessian( const InputPointType & ipp,
  SpatialHessianType & sh ) const
{
  if( this->GetNumberOfParameters() == 0 )
  {
    for( unsigned int i = 0; i < sh.Size(); ++i )
    {
      sh[ i ].Fill( 0.0 );
    }
    return;
  }

  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  int lidx = 0;
  PointToLabel( ipp, lidx );
  if( lidx == 0 )
  {
    for( unsigned int i = 0; i < sh.Size(); ++i )
    {
      sh[ i ].Fill( 0.0 );
    }
    return;
  }

  SpatialHessianType nsh, lsh;
  m_Trans[ 0 ]->GetSpatialHessian( ipp, nsh );
  m_Trans[ lidx ]->GetSpatialHessian( ipp, lsh );
  for( unsigned i = 0; i < SpaceDimension; ++i )
  {
    for( unsigned j = 0; j < SpaceDimension; ++j )
    {
      for( unsigned k = 0; k < SpaceDimension; ++k )
      {
        sh[ i ][ j ][ k ] = lsh[ i ][ j ][ k ] + nsh[ i ][ j ][ k ];
      }
    }
  }
}


template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::GetJacobianOfSpatialJacobian(
  const InputPointType & ipp,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  itkExceptionMacro( << "ERROR: GetJacobianOfSpatialJacobian() not yet implemented "
                     << "in the MultiBSplineDeformableTransformWithNormal class." );
} // end GetJacobianOfSpatialJacobian()


template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::GetJacobianOfSpatialJacobian(
  const InputPointType & ipp,
  SpatialJacobianType & sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  if( this->GetNumberOfParameters() == 0 )
  {
    jsj.resize( 0 );
    nonZeroJacobianIndices.resize( 0 );
    return;
  }

  // Initialize
  const unsigned int nnzji = this->GetNumberOfNonZeroJacobianIndices();
  jsj.resize( nnzji );

  // This implements a sparse version of the Jacobian.
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  int lidx = 0;
  PointToLabel( ipp, lidx );

  // Convert the physical point to a continuous index, which
  // is needed for the 'Evaluate()' functions below.
  typename TransformType::ContinuousIndexType cindex;
  m_Trans[ lidx ]->TransformPointToContinuousGridIndex( ipp, cindex );

  if( lidx == 0 || !m_Trans[ lidx ]->InsideValidRegion( cindex ) )
  {
    sj.SetIdentity();
    for( unsigned int i = 0; i < jsj.size(); ++i )
    {
      jsj[ i ].Fill( 0.0 );
    }
    nonZeroJacobianIndices.resize( nnzji );
    for( unsigned int i = 0; i < nnzji; ++i )
    {
      nonZeroJacobianIndices[ i ] = i;
    }
    return;
  }

  SpatialJacobianType           nsj, lsj;
  JacobianOfSpatialJacobianType njsj, ljsj;

  // nzji should be the same so keep only one
  m_Trans[ 0 ]->GetJacobianOfSpatialJacobian( ipp, nsj, njsj, nonZeroJacobianIndices );
  m_Trans[ lidx ]->GetJacobianOfSpatialJacobian( ipp, lsj, ljsj, nonZeroJacobianIndices );

  typedef typename ImageBaseType::PixelContainer BaseContainer;
  const BaseContainer & bases = *m_LocalBases->GetPixelContainer();

  const unsigned nweights = this->GetNumberOfWeights();
  for( unsigned i = 0; i < nweights; ++i )
  {
    VectorType tmp = bases[ nonZeroJacobianIndices[ i ] ][ 0 ];
    for( unsigned j = 0; j < SpaceDimension; ++j )
    {
      for( unsigned k = 0; k < SpaceDimension; ++k )
      {
        jsj[ j ][ i ][ k ] = tmp[ j ] * njsj[ j ][ i + j * nweights ][ k ];
      }
    }

    for( unsigned d = 1; d < SpaceDimension; ++d )
    {
      tmp = bases[ nonZeroJacobianIndices[ i ] ][ d ];
      for( unsigned j = 0; j < SpaceDimension; ++j )
      {
        for( unsigned k = 0; k < SpaceDimension; ++k )
        {
          jsj[ j ][ i + d * nweights ][ k ] = tmp[ j ] * ljsj[ j ][ i + j * nweights ][ k ];
        }
      }
    }
    sj = nsj + lsj;
  }

  // move non zero indices to match label positions
  if( lidx > 1 )
  {
    unsigned to_add = ( lidx - 1 ) * m_Trans[ 0 ]->GetNumberOfParametersPerDimension() * ( SpaceDimension - 1 );
    for( unsigned i = 0; i < nweights; ++i )
    {
      for( unsigned d = 1; d < SpaceDimension; ++d )
      {
        nonZeroJacobianIndices[ d * nweights + i ] += to_add;
      }
    }
  }
}


template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
void
MultiBSplineDeformableTransformWithNormal< TScalarType, NDimensions, VSplineOrder >
::GetJacobianOfSpatialHessian(
  const InputPointType & ipp,
  SpatialHessianType & sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  if( this->GetNumberOfParameters() == 0 )
  {
    jsh.resize( 0 );
    nonZeroJacobianIndices.resize( 0 );
    return;
  }

  // Initialize
  const unsigned int nnzji = this->GetNumberOfNonZeroJacobianIndices();
  jsh.resize( nnzji );

  // This implements a sparse version of the Jacobian.
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if( this->m_InputParametersPointer == NULL )
  {
    itkExceptionMacro( << "Cannot compute Jacobian: parameters not set" );
  }

  int lidx = 0;
  PointToLabel( ipp, lidx );

  // Convert the physical point to a continuous index, which
  // is needed for the 'Evaluate()' functions below.
  typename TransformType::ContinuousIndexType cindex;
  m_Trans[ lidx ]->TransformPointToContinuousGridIndex( ipp, cindex );

  if( lidx == 0 || !m_Trans[ lidx ]->InsideValidRegion( cindex ) )
  {
    // Return some dummy
    for( unsigned int i = 0; i < jsh.size(); ++i )
    {
      for( unsigned int j = 0; j < jsh[ i ].Size(); ++j )
      {
        jsh[ i ][ j ].Fill( 0.0 );
      }
    }
    for( unsigned int i = 0; i < sh.Size(); ++i )
    {
      sh[ i ].Fill( 0.0 );
    }
    nonZeroJacobianIndices.resize( nnzji );
    for( unsigned int i = 0; i < nnzji; ++i )
    {
      nonZeroJacobianIndices[ i ] = i;
    }
    return;
  }

  SpatialHessianType           nsh, lsh;
  JacobianOfSpatialHessianType njsh, ljsh;

  // nzji should be the same so keep only one
  m_Trans[ 0 ]->GetJacobianOfSpatialHessian( ipp, nsh, njsh, nonZeroJacobianIndices );
  m_Trans[ lidx ]->GetJacobianOfSpatialHessian( ipp, lsh, ljsh, nonZeroJacobianIndices );

  typedef typename ImageBaseType::PixelContainer BaseContainer;
  const BaseContainer & bases = *m_LocalBases->GetPixelContainer();

  const unsigned nweights = this->GetNumberOfWeights();
  for( unsigned i = 0; i < nweights; ++i )
  {
    VectorType tmp = bases[ nonZeroJacobianIndices[ i ] ][ 0 ];
    for( unsigned j = 0; j < SpaceDimension; ++j )
    {
      for( unsigned k = 0; k < SpaceDimension; ++k )
      {
        for( unsigned l = 0; l < SpaceDimension; ++l )
        {
          jsh[ j ][ i ][ k ][ l ] = tmp[ j ] * njsh[ j ][ i + j * nweights ][ k ][ l ];
        }
      }
    }

    for( unsigned l = 1; l <= m_NbLabels; ++l )
    {
      VectorType tmp = bases[ nonZeroJacobianIndices[ i ] ][ l ];
      for( unsigned j = 0; j < SpaceDimension; ++j )
      {
        for( unsigned k = 0; k < SpaceDimension; ++k )
        {
          for( unsigned l = 0; l < SpaceDimension; ++l )
          {
            jsh[ j ][ i + l * nweights ][ k ][ l ] = tmp[ j ] * ljsh[ j ][ i + j * nweights ][ k ][ l ];
          }
        }
      }
    }
  }

  for( unsigned i = 0; i < SpaceDimension; ++i )
  {
    for( unsigned j = 0; j < SpaceDimension; ++j )
    {
      for( unsigned k = 0; k < SpaceDimension; ++k )
      {
        sh[ i ][ j ][ k ] = lsh[ i ][ j ][ k ] + nsh[ i ][ j ][ k ];
      }
    }
  }

  // move non zero indices to match label positions
  if( lidx > 1 )
  {
    unsigned to_add = ( lidx - 1 ) * m_Trans[ 0 ]->GetNumberOfParametersPerDimension() * ( SpaceDimension - 1 );
    for( unsigned i = 0; i < nweights; ++i )
    {
      for( unsigned d = 1; d < SpaceDimension; ++d )
      {
        nonZeroJacobianIndices[ d * nweights + i ] += to_add;
      }
    }
  }
}


} // end namespace itk

#endif // end #ifndef __itkMultiBSplineDeformableTransformWithNormal_hxx
