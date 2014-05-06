/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef _itkCombinationImageToImageMetric_hxx
#define _itkCombinationImageToImageMetric_hxx

#include "itkCombinationImageToImageMetric.h"
#include "itkTimeProbe.h"
#include "itkMath.h"


/** Macros to reduce some copy-paste work.
 * These macros provide the implementation of
 * all Set/GetFixedImage, Set/GetInterpolator etc methods
 *
 * The macros are undef'ed at the end of this file
 */

/** For setting objects, implement two methods */
#define itkImplementationSetObjectMacro2( _name, _type1, _type2 ) \
  template< class TFixedImage, class TMovingImage > \
  void \
  CombinationImageToImageMetric< TFixedImage, TMovingImage > \
  ::Set##_name( _type1 _type2 * _arg, unsigned int pos ) \
  { \
    if( pos == 0 ) \
    { \
      this->Superclass::Set##_name( _arg ); \
    } \
    ImageMetricType * testPtr1 \
      = dynamic_cast< ImageMetricType * >( this->GetMetric( pos ) ); \
    PointSetMetricType * testPtr2 \
      = dynamic_cast< PointSetMetricType * >( this->GetMetric( pos ) ); \
    if( testPtr1 ) \
    { \
      testPtr1->Set##_name( _arg ); \
    } \
    else if( testPtr2 ) \
    { \
      testPtr2->Set##_name( _arg ); \
    } \
  } \
  template< class TFixedImage, class TMovingImage > \
  void \
  CombinationImageToImageMetric< TFixedImage, TMovingImage > \
  ::Set##_name( _type1 _type2 * _arg ) \
  { \
    for( unsigned int i = 0; i < this->GetNumberOfMetrics(); i++ ) \
    { \
      this->Set##_name( _arg, i ); \
    } \
  } // comments for allowing ; after calling the macro

#define itkImplementationSetObjectMacro1( _name, _type1, _type2 ) \
  template< class TFixedImage, class TMovingImage > \
  void \
  CombinationImageToImageMetric< TFixedImage, TMovingImage > \
  ::Set##_name( _type1 _type2 * _arg, unsigned int pos ) \
  { \
    if( pos == 0 ) \
    { \
      this->Superclass::Set##_name( _arg ); \
    } \
    ImageMetricType * testPtr1 \
      = dynamic_cast< ImageMetricType * >( this->GetMetric( pos ) ); \
    if( testPtr1 ) \
    { \
      testPtr1->Set##_name( _arg ); \
    } \
  } \
  template< class TFixedImage, class TMovingImage > \
  void \
  CombinationImageToImageMetric< TFixedImage, TMovingImage > \
  ::Set##_name( _type1 _type2 * _arg ) \
  { \
    for( unsigned int i = 0; i < this->GetNumberOfMetrics(); i++ ) \
    { \
      this->Set##_name( _arg, i ); \
    } \
  } // comments for allowing ; after calling the macro

/** for getting const object, implement one method */
#define itkImplementationGetConstObjectMacro1( _name, _type ) \
  template< class TFixedImage, class TMovingImage > \
  const typename CombinationImageToImageMetric< TFixedImage, TMovingImage >:: \
  _type * CombinationImageToImageMetric< TFixedImage, TMovingImage > \
  ::Get##_name( unsigned int pos ) const \
  { \
    const ImageMetricType * testPtr1 \
      = dynamic_cast< const ImageMetricType * >( this->GetMetric( pos ) ); \
    if( testPtr1 ) \
    { \
      return testPtr1->Get##_name(); \
    } \
    else \
    { \
      return 0; \
    } \
  } // comments for allowing ; after calling the macro

#define itkImplementationGetConstObjectMacro2( _name, _type ) \
  template< class TFixedImage, class TMovingImage > \
  const typename CombinationImageToImageMetric< TFixedImage, TMovingImage >:: \
  _type * CombinationImageToImageMetric< TFixedImage, TMovingImage > \
  ::Get##_name( unsigned int pos ) const \
  { \
    const ImageMetricType * testPtr1 \
      = dynamic_cast< const ImageMetricType * >( this->GetMetric( pos ) ); \
    const PointSetMetricType * testPtr2 \
      = dynamic_cast< const PointSetMetricType * >( this->GetMetric( pos ) ); \
    if( testPtr1 ) \
    { \
      return testPtr1->Get##_name(); \
    } \
    else if( testPtr2 ) \
    { \
      return testPtr2->Get##_name(); \
    } \
    else \
    { \
      return 0; \
    } \
  } // comments for allowing ; after calling the macro

namespace itk
{

itkImplementationSetObjectMacro2( Transform, , TransformType );
itkImplementationSetObjectMacro1( Interpolator, , InterpolatorType );
itkImplementationSetObjectMacro2( FixedImageMask, , FixedImageMaskType );
itkImplementationSetObjectMacro2( MovingImageMask, , MovingImageMaskType );
itkImplementationSetObjectMacro1( FixedImage, const, FixedImageType );
itkImplementationSetObjectMacro1( MovingImage, const, MovingImageType );

itkImplementationGetConstObjectMacro2( Transform, TransformType );
itkImplementationGetConstObjectMacro1( Interpolator, InterpolatorType );
itkImplementationGetConstObjectMacro2( FixedImageMask, FixedImageMaskType );
itkImplementationGetConstObjectMacro2( MovingImageMask, MovingImageMaskType );
itkImplementationGetConstObjectMacro1( FixedImage, FixedImageType );
itkImplementationGetConstObjectMacro1( MovingImage, MovingImageType );

/**
 * ********************* Constructor ****************************
 */

template< class TFixedImage, class TMovingImage >
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::CombinationImageToImageMetric()
{
  this->m_NumberOfMetrics    = 0;
  this->m_UseRelativeWeights = false;
  this->ComputeGradientOff();

  this->m_UseMultiThread = true;

} // end Constructor


/**
 * ********************* PrintSelf ****************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  /** Call the superclass' PrintSelf. */
  Superclass::PrintSelf( os, indent );

  /** Add debugging information. */
  os << "NumberOfMetrics: " << this->m_NumberOfMetrics << std::endl;
  for( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
  {
    os << "Metric " << i << ":\n";
    os << indent << "MetricPointer: " << this->m_Metrics[ i ].GetPointer() << "\n";
    os << indent << "MetricWeight: " << this->m_MetricWeights[ i ] << "\n";
    os << indent << "MetricRelativeWeight: " << this->m_MetricRelativeWeights[ i ] << "\n";
    os << indent << "UseRelativeWeights: " << ( this->m_UseRelativeWeights ? "true\n" : "false\n" );
    os << indent << "MetricValue: " << this->m_MetricValues[ i ] << "\n";
    os << indent << "MetricDerivativesMagnitude: "  << this->m_MetricDerivativesMagnitude[ i ] << "\n";
    os << indent << "UseMetric: " << ( this->m_UseMetric[ i ] ? "true\n" : "false\n" );
    os << indent << "MetricComputationTime: " << this->m_MetricComputationTime[ i ] << "\n";
  }

} // end PrintSelf()


/**
 * ******************** SetFixedImageRegion ************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::SetFixedImageRegion( const FixedImageRegionType _arg, unsigned int pos )
{
  if( pos == 0 )
  {
    this->Superclass::SetFixedImageRegion( _arg );
  }
  ImageMetricType * testPtr = dynamic_cast< ImageMetricType * >(
    this->GetMetric( pos ) );
  if( testPtr )
  {
    testPtr->SetFixedImageRegion( _arg );
  }

} // end SetFixedImageRegion()


/**
 * ******************** SetFixedImageRegion ************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::SetFixedImageRegion( const FixedImageRegionType _arg )
{
  for( unsigned int i = 0; i < this->GetNumberOfMetrics(); i++ )
  {
    this->SetFixedImageRegion( _arg, i );
  }

} // end SetFixedImageRegion()


/**
 * ******************** GetFixedImageRegion ************************
 */

template< class TFixedImage, class TMovingImage >
const typename CombinationImageToImageMetric< TFixedImage, TMovingImage >
::FixedImageRegionType
& CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetFixedImageRegion( unsigned int pos ) const
{
  const ImageMetricType * testPtr
    = dynamic_cast< const ImageMetricType * >( this->GetMetric( pos ) );
  if( testPtr )
  {
    return testPtr->GetFixedImageRegion();
  }
  else
  {
    return this->m_NullFixedImageRegion;
  }

} // end GetFixedImageRegion()

/**
 * ********************* SetNumberOfMetrics ****************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::SetNumberOfMetrics( unsigned int count )
{
  if( count != this->m_Metrics.size() )
  {
    this->m_NumberOfMetrics = count;
    this->m_Metrics.resize( count );
    this->m_MetricWeights.resize( count );
    this->m_MetricRelativeWeights.resize( count );
    this->m_UseMetric.resize( count );
    this->m_MetricValues.resize( count );
    this->m_MetricDerivatives.resize( count );
    this->m_MetricDerivativesMagnitude.resize( count );
    this->m_MetricComputationTime.resize( count );
    this->Modified();
  }

} // end SetNumberOfMetrics()


/**
 * ********************* SetMetric ****************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::SetMetric( SingleValuedCostFunctionType * metric, unsigned int pos )
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    this->SetNumberOfMetrics( pos + 1 );
  }

  if( metric != this->m_Metrics[ pos ] )
  {
    this->m_Metrics[ pos ] = metric;
    this->Modified();
  }

} // end SetMetric()


/**
 * ********************* GetMetric ****************************
 */

template< class TFixedImage, class TMovingImage >
typename CombinationImageToImageMetric< TFixedImage, TMovingImage >
::SingleValuedCostFunctionType
* CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetMetric( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    return 0;
  }
  else
  {
    return this->m_Metrics[ pos ];
  }

} // end GetMetric()

/**
 * ********************* SetMetricWeight ****************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::SetMetricWeight( double weight, unsigned int pos )
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    this->SetNumberOfMetrics( pos + 1 );
  }

  if( weight != this->m_MetricWeights[ pos ] )
  {
    this->m_MetricWeights[ pos ] = weight;
    this->Modified();
  }

} // end SetMetricWeight()


/**
 * ********************* GetMetricWeight ****************************
 */

template< class TFixedImage, class TMovingImage >
double
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetMetricWeight( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    return 0.0;
  }
  else
  {
    return this->m_MetricWeights[ pos ];
  }

} // end GetMetricWeight()


/**
 * ********************* SetMetricRelativeWeight ****************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::SetMetricRelativeWeight( double weight, unsigned int pos )
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    this->SetNumberOfMetrics( pos + 1 );
  }

  if( weight != this->m_MetricRelativeWeights[ pos ] )
  {
    this->m_MetricRelativeWeights[ pos ] = weight;
    this->Modified();
  }

} // end SetMetricRelativeWeight()


/**
 * ********************* GetMetricRelativeWeight ****************************
 */

template< class TFixedImage, class TMovingImage >
double
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetMetricRelativeWeight( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    return 0.0;
  }
  else
  {
    return this->m_MetricRelativeWeights[ pos ];
  }

} // end GetMetricRelativeWeight()


/**
 * ********************* SetUseMetric ****************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::SetUseMetric( const bool use, const unsigned int pos )
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    this->SetNumberOfMetrics( pos + 1 );
  }

  if( use != this->m_UseMetric[ pos ] )
  {
    this->m_UseMetric[ pos ] = use;
    this->Modified();
  }

} // end SetUseMetric()


/**
 * ********************* SetUseAllMetrics ****************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::SetUseAllMetrics( void )
{
  for( unsigned int pos = 0; pos < this->GetNumberOfMetrics(); ++pos )
  {
    if( !this->m_UseMetric[ pos ] )
    {
      this->m_UseMetric[ pos ] = true;
      this->Modified();
    }
  }

} // end SetUseAllMetrics()


/**
 * ********************* GetUseMetric ****************************
 */

template< class TFixedImage, class TMovingImage >
bool
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetUseMetric( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    return false;
  }
  else
  {
    return this->m_UseMetric[ pos ];
  }

} // end GetUseMetric()


/**
 * ********************* GetMetricValue ****************************
 */

template< class TFixedImage, class TMovingImage >
typename CombinationImageToImageMetric< TFixedImage, TMovingImage >::MeasureType
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetMetricValue( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    return 0.0;
  }
  else
  {
    return this->m_MetricValues[ pos ];
  }

} // end GetMetricValue()


/**
 * ********************* GetMetricDerivative ****************************
 */

template< class TFixedImage, class TMovingImage >
const typename CombinationImageToImageMetric< TFixedImage, TMovingImage >
::DerivativeType
& CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetMetricDerivative( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    return this->m_NullDerivative;
  }
  else
  {
    return this->m_MetricDerivatives[ pos ];
  }

} // end GetMetricDerivative()

/**
 * ********************* GetMetricDerivativeMagnitude ****************************
 */

template< class TFixedImage, class TMovingImage >
double
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetMetricDerivativeMagnitude( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    return 0.0;
  }
  else
  {
    return this->m_MetricDerivativesMagnitude[ pos ];
  }

} // end GetMetricDerivativeMagnitude()


/**
 * ********************* GetMetricComputationTime ****************************
 */

template< class TFixedImage, class TMovingImage >
std::size_t
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetMetricComputationTime( unsigned int pos ) const
{
  if( pos >= this->GetNumberOfMetrics() )
  {
    return 0;
  }
  else
  {
    return this->m_MetricComputationTime[ pos ];
  }

} // end GetMetricComputationTime()


/**
 * **************** GetNumberOfPixelsCounted ************************
 */

template< class TFixedImage, class TMovingImage >
const SizeValueType &
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetNumberOfPixelsCounted( void ) const
{
  unsigned long sum = 0;
  for( unsigned int i = 0; i < this->GetNumberOfMetrics(); ++i )
  {
    const ImageMetricType * testPtr
      = dynamic_cast< const ImageMetricType * >( this->GetMetric( i ) );
    if( testPtr )
    {
      sum += testPtr->GetNumberOfPixelsCounted();
    }
  }

  this->m_NumberOfPixelsCounted = sum;
  return this->m_NumberOfPixelsCounted;

} // end GetNumberOfPixelsCounted()


/**
 * ********************* Initialize ****************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::Initialize( void ) throw ( ExceptionObject )
{
  /** Check if transform, interpolator have been set. Effectively this
   * method checks if the first sub metric is set up completely.
   * This implicitly means that the first sub metric is an ImageToImageMetric,
   * which is a reasonable demand.
   */
  this->Superclass::Initialize();

  /** Check if at least one (image)metric is provided */
  if( this->GetNumberOfMetrics() == 0 )
  {
    itkExceptionMacro( << "At least one metric should be set!" );
  }

  /** Call Initialize for all metrics. */
  for( unsigned int i = 0; i < this->GetNumberOfMetrics(); i++ )
  {
    SingleValuedCostFunctionType * costfunc = this->GetMetric( i );
    if( !costfunc )
    {
      itkExceptionMacro( << "Metric " << i << " has not been set!" );
    }
    ImageMetricType *    testPtr1 = dynamic_cast< ImageMetricType * >( this->GetMetric( i ) );
    PointSetMetricType * testPtr2 = dynamic_cast< PointSetMetricType * >( this->GetMetric( i ) );
    if( testPtr1 )
    {
      // The NumberOfThreadsPerMetric is changed after Initialize() so we save it before and then
      // set it on.
      unsigned nrOfThreadsPerMetric = this->GetNumberOfThreads();
      testPtr1->Initialize();
      testPtr1->SetNumberOfThreads( nrOfThreadsPerMetric );
    }
    else if( testPtr2 )
    {
      testPtr2->Initialize();
    }
  }

} // end Initialize()


/**
 * ******************* InitializeThreadingParameters *******************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::InitializeThreadingParameters( void ) const
{
  /** Initialize the derivatives. */
  for( ThreadIdType i = 0; i < this->GetNumberOfMetrics(); ++i )
  {
    this->m_MetricDerivatives[ i ].SetSize( this->GetNumberOfParameters() );
  }
} // end InitializeThreadingParameters()


/**
 * ******************* GetFinalMetricWeight *******************
 */

template< class TFixedImage, class TMovingImage >
double
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetFinalMetricWeight( unsigned int pos ) const
{
  double weight = 1.0;
  if( !this->m_UseRelativeWeights )
  {
    weight = this->m_MetricWeights[ pos ];
  }
  else
  {
    /** The relative weight of metric i is such that the
     * magnitude of the derivative of metric i is rescaled
     * to be a fraction of that of metric 0; the fraction is
     * defined by the fraction of the two relative weights.
     * Note that this weight is different in each iteration.
     */
    if( this->m_MetricDerivativesMagnitude[ pos ] > 1e-10 )
    {
      weight = this->m_MetricRelativeWeights[ pos ]
        * this->m_MetricDerivativesMagnitude[ 0 ]
        / this->m_MetricDerivativesMagnitude[ pos ];
    }
  }

  return weight;
} // end GetFinalMetricWeight()


/**
 * ********************* GetValue ****************************
 */

template< class TFixedImage, class TMovingImage >
typename CombinationImageToImageMetric< TFixedImage, TMovingImage >::MeasureType
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetValue( const ParametersType & parameters ) const
{
  /** Initialise. */
  MeasureType measure = NumericTraits< MeasureType >::Zero;

  /** Compute, store and combine all metric values. */
  for( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
  {
    /** Time the computation per metric. */
    itk::TimeProbe timer;
    timer.Start();

    /** Compute ... */
    MeasureType tmpValue = this->m_Metrics[ i ]->GetValue( parameters );
    timer.Stop();

    /** store ... */
    this->m_MetricValues[ i ] = tmpValue;
    this->m_MetricComputationTime[ i ]
      = Math::Round< std::size_t, double >( timer.GetMean() * 1000.0 );

    /** and combine. */
    if( this->m_UseMetric[ i ] )
    {
      if( !this->m_UseRelativeWeights )
      {
        measure += this->m_MetricWeights[ i ] * this->m_MetricValues[ i ];
      }
      else
      {
        /** The relative weight of metric i is such that the
         * value of metric i is rescaled
         * to be a fraction of that of metric 0; the fraction is
         * defined by the fraction of the two relative weights.
         * Note that this weight is different in each iteration.
         */
        double weight = 1.0;
        if( this->m_MetricValues[ i ] > 1e-10 )
        {
          weight = this->m_MetricRelativeWeights[ i ]
            * this->m_MetricValues[ 0 ]
            / this->m_MetricValues[ i ];
          measure += weight * this->m_MetricValues[ i ];
        }
      }
    }
  }

  /** Return a value. */
  return measure;

} // end GetValue()


/**
 * ********************* GetDerivative ****************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetDerivative( const ParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** Initialise. */
  DerivativeType tmpDerivative = DerivativeType( this->GetNumberOfParameters() );
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< MeasureType >::Zero );

  /** Compute, store and combine all metric derivatives. */
  for( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
  {
    /** Time the computation per metric. */
    itk::TimeProbe timer;
    timer.Start();

    /** Compute ... */
    tmpDerivative.Fill( NumericTraits< MeasureType >::Zero );
    this->m_Metrics[ i ]->GetDerivative( parameters, tmpDerivative );
    timer.Stop();

    /** store ... */
    this->m_MetricDerivatives[ i ]          = tmpDerivative;
    this->m_MetricDerivativesMagnitude[ i ] = tmpDerivative.magnitude();
    this->m_MetricComputationTime[ i ]
      = Math::Round< std::size_t, double >( timer.GetMean() * 1000.0 );

    /** and combine. */
    if( this->m_UseMetric[ i ] )
    {
      if( !this->m_UseRelativeWeights )
      {
        derivative += this->m_MetricWeights[ i ] * this->m_MetricDerivatives[ i ];
      }
      else
      {
        /** The relative weight of metric i is such that the
         * magnitude of the derivative of metric i is rescaled
         * to be a fraction of that of metric 0; the fraction is
         * defined by the fraction of the two relative weights.
         * Note that this weight is different in each iteration.
         */
        double weight = 1.0;
        if( this->m_MetricDerivativesMagnitude[ i ] > 1e-10 )
        {
          weight = this->m_MetricRelativeWeights[ i ]
            * this->m_MetricDerivativesMagnitude[ 0 ]
            / this->m_MetricDerivativesMagnitude[ i ];
          derivative += weight * this->m_MetricDerivatives[ i ];
        }
      }
    }
  }

} // end GetDerivative()


/**
 * ********************* GetValueAndDerivative ****************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  /** Declare timer and multi-threader. */
  itk::TimeProbe timer;
  typename ThreaderType::Pointer local_threader = ThreaderType::New();

  /** This function must be called before the multi-threaded code.
   * It calls all the non thread-safe stuff.
   */
  for( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
  {
    ImageMetricType *    testPtr1 = dynamic_cast< ImageMetricType * >( this->GetMetric( i ) );
    PointSetMetricType * testPtr2 = dynamic_cast< PointSetMetricType * >( this->GetMetric( i ) );
    if( testPtr1 )
    {
      testPtr1->SetUseMetricSingleThreaded( true );
      testPtr1->BeforeThreadedGetValueAndDerivative( parameters );
      testPtr1->SetUseMetricSingleThreaded( false );
    }
    if( testPtr2 )
    {
      testPtr2->SetUseMetricSingleThreaded( true );
      testPtr2->BeforeThreadedGetValueAndDerivative( parameters );
      testPtr2->SetUseMetricSingleThreaded( false );
    }
  }

  /** Initialize some threading related parameters. */
  this->InitializeThreadingParameters();

  /** Decide whether or not to use multi-threading.
   * If the global maximum number of threads is smaller than the number
   * of metrics, then setting the number of threads to the number of
   * threads will result in a bug: all metrics after the max will not be
   * run. For now we decide to run this metric single-threadedly in that case.
   *
   * Actually we should sum the number of threads of each metric and check that
   * it does not surpass the global maximum. But for now we don't.
   */
  bool useMultiThread = this->m_UseMultiThread;
  if( MultiThreader::GetGlobalMaximumNumberOfThreads() < this->m_NumberOfMetrics )
  {
    useMultiThread = false;
  }

  // For now we disable multi-threading of the combo metric, since the MacMini build
  // segfaults with it. Have to investigate this issue later
  useMultiThread = false;

  /** Compute all metric values and derivatives, single-threadedly. */
  if( !useMultiThread )
  {
    for( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      /** Compute ... */
      timer.Reset();
      timer.Start();
      this->m_Metrics[ i ]->GetValueAndDerivative( parameters,
        this->m_MetricValues[ i ], this->m_MetricDerivatives[ i ] );
      timer.Stop();

      /** Store computation time. */
      this->m_MetricComputationTime[ i ]
        = Math::Round< std::size_t, double >( timer.GetMean() * 1000.0 );
    }
  }
  /** Compute all metric values and derivatives, multi-threadedly. */
  else
  {
    /** Setup struct with multi-threading information. */
    MultiThreaderComboMetricsType * temp_c = new MultiThreaderComboMetricsType;
    temp_c->st_MetricsIterator           = this->m_Metrics;
    temp_c->st_MetricDerivativesIterator = this->m_MetricDerivatives.begin();
    temp_c->st_MetricValuesIterator      = this->m_MetricValues.begin();
    temp_c->st_MetricComputationTime.resize( this->m_NumberOfMetrics, 0 );
    temp_c->st_Parameters = const_cast< ParametersType * >( &parameters );

    /** GetValueAndDerivative */
    local_threader->SetNumberOfThreads( this->m_NumberOfMetrics );
    local_threader->SetSingleMethod( GetValueAndDerivativeComboThreaderCallback, temp_c );
    local_threader->SingleMethodExecute();

    /** Store computation time. */
    for( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      this->m_MetricComputationTime[ i ] = temp_c->st_MetricComputationTime[ i ];
    }

    delete temp_c;
  }

  /** Compute the derivative magnitude, single-threadedly. */
  // I get random segfaults on the Linux or Mac systems when this is enabled.
  // For now disable multi-threading and force single-threaded:
  //if( !this->m_UseMultiThread )
  if( true )
  {
    for( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      this->m_MetricDerivativesMagnitude[ i ] = this->m_MetricDerivatives[ i ].magnitude();
    }
  }
  /** Compute the derivative magnitude, multi-threadedly. */
  else
  {
    /** Setup struct with multi-threading information. */
    ThreadIdType                         numberOfThreads = this->GetNumberOfThreads();
    MultiThreaderCombineDerivativeType * temp_m          = new MultiThreaderCombineDerivativeType;
    temp_m->st_ThisComboMetric = const_cast< Self * >( this );
    temp_m->st_DerivativesSumOfSquares.resize( numberOfThreads * this->m_NumberOfMetrics );

    /** Compute derivatives magnitude multi-threadedly. */
    local_threader->SetNumberOfThreads( numberOfThreads );
    local_threader->SetSingleMethod( ComputeDerivativesMagnitudeThreaderCallback, temp_m );
    local_threader->SingleMethodExecute();

    /** Gather the results. */
    for( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
    {
      double mag = 0.0;
      for( unsigned int j = 0; j < numberOfThreads; j++ )
      {
        mag += temp_m->st_DerivativesSumOfSquares[ i * numberOfThreads + j ];
      }
      this->m_MetricDerivativesMagnitude[ i ] = vcl_sqrt( mag );
    }

    delete temp_m;
  }

  /** Combine the metric values, single-threadedly. */
  value = NumericTraits< MeasureType >::Zero;
  for( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
  {
    if( this->m_UseMetric[ i ] )
    {
      double weight = this->GetFinalMetricWeight( i );
      value += weight * this->m_MetricValues[ i ];
    } // end if m_UseMetric[i]
  }   // end of combine metrics

  /** Combine the metric derivatives, single-threadedly. */
  // I get random segfaults on the linux or Mac systems when this is enabled.
  // For now disable multi-threading and force single-threaded:
  //if( !this->m_UseMultiThread )
  if( true )
  {
    /** The first derivative. */
    if( this->m_UseMetric[ 0 ] )
    {
      double weight = this->GetFinalMetricWeight( 0 );
      derivative = weight * this->m_MetricDerivatives[ 0 ];
    }
    else
    {
      derivative.Fill( 0 );
    }

    /** The remaining derivatives. */
    for( unsigned int i = 1; i < this->m_NumberOfMetrics; i++ )
    {
      /** and combine. */
      if( this->m_UseMetric[ i ] )
      {
        double weight = this->GetFinalMetricWeight( i );
        derivative += weight * this->m_MetricDerivatives[ i ];
      } // end if m_UseMetric[i]
    }   // end of combine metrics
  }
  /** Combine the derivatives, multi-threadedly. */
  else
  {
    /** Setup struct with multi-threading information. */
    ThreadIdType                         numberOfThreads = this->GetNumberOfThreads();
    MultiThreaderCombineDerivativeType * temp_d          = new MultiThreaderCombineDerivativeType;
    temp_d->st_ThisComboMetric = const_cast< Self * >( this );
    temp_d->st_Derivative      = &derivative[ 0 ];

    /** Combine derivatives */
    local_threader->SetNumberOfThreads( numberOfThreads );
    local_threader->SetSingleMethod( CombineDerivativesThreaderCallback, temp_d );
    local_threader->SingleMethodExecute();

    delete temp_d;
  }

} // end GetValueAndDerivative()


/**
 * **************** GetValueAndDerivativeThreaderCallback *******
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivativeComboThreaderCallback( void * arg )
{
  ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
  ThreadIdType     threadID   = infoStruct->ThreadID;

  MultiThreaderComboMetricsType * temp
    = static_cast< MultiThreaderComboMetricsType * >( infoStruct->UserData );

  itk::TimeProbe timer;
  timer.Start();
  temp->st_MetricsIterator[ threadID ]->GetValueAndDerivative(
    *temp->st_Parameters,
    temp->st_MetricValuesIterator[ threadID ],
    temp->st_MetricDerivativesIterator[ threadID ] );
  timer.Stop();
  temp->st_MetricComputationTime[ threadID ]
    = Math::Round< std::size_t, double >( timer.GetMean() * 1000.0 );

  return ITK_THREAD_RETURN_VALUE;

} // end GetValueAndDerivativeThreaderCallback()


/**
 *********** ComputeDerivativesMagnitudeThreaderCallback *************
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::ComputeDerivativesMagnitudeThreaderCallback( void * arg )
{
  ThreadInfoType * infoStruct  = static_cast< ThreadInfoType * >( arg );
  ThreadIdType     threadId    = infoStruct->ThreadID;
  ThreadIdType     nrOfThreads = infoStruct->NumberOfThreads;

  MultiThreaderCombineDerivativeType * temp
    = static_cast< MultiThreaderCombineDerivativeType * >( infoStruct->UserData );

  /** Get references to member variables. */
  const unsigned int numberOfParameters = temp->st_ThisComboMetric->GetNumberOfParameters();
  const unsigned int numberOfMetrics    = temp->st_ThisComboMetric->GetNumberOfMetrics();

  /** Determine compute range of this thread. */
  const unsigned int subSize = static_cast< unsigned int >(
    vcl_ceil( static_cast< double >( numberOfParameters ) / static_cast< double >( nrOfThreads ) ) );
  unsigned int jmin = threadId * subSize;
  unsigned int jmax = ( threadId + 1 ) * subSize;
  jmax = ( jmax > numberOfParameters ) ? numberOfParameters : jmax;

  /** Compute the sum of squares within compute range. */
  double derivativeValue = 0.0;
  for( unsigned int i = 0; i < numberOfMetrics; i++ )
  {
    if( temp->st_ThisComboMetric->m_UseMetric[ i ] )
    {
      const DerivativeType & derivative   = temp->st_ThisComboMetric->m_MetricDerivatives[ i ];
      double                 sumOfSquares = 0.0;
      for( unsigned int j = jmin; j < jmax; j++ )
      {
        derivativeValue = derivative[ j ];
        sumOfSquares   += derivativeValue * derivativeValue;
      }
      temp->st_DerivativesSumOfSquares[ i * nrOfThreads + threadId ] = sumOfSquares;
    }
  }

  return ITK_THREAD_RETURN_VALUE;

} // end ComputeDerivativesMagnitudeThreaderCallback()


/**
 *********** CombineDerivativesThreaderCallback *************
 */

template< class TFixedImage, class TMovingImage >
ITK_THREAD_RETURN_TYPE
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::CombineDerivativesThreaderCallback( void * arg )
{
  ThreadInfoType * infoStruct  = static_cast< ThreadInfoType * >( arg );
  ThreadIdType     threadId    = infoStruct->ThreadID;
  ThreadIdType     nrOfThreads = infoStruct->NumberOfThreads;

  MultiThreaderCombineDerivativeType * temp
    = static_cast< MultiThreaderCombineDerivativeType * >( infoStruct->UserData );

  /** Get references to member variables. */
  const unsigned int numberOfParameters = temp->st_ThisComboMetric->GetNumberOfParameters();
  const unsigned int numberOfMetrics    = temp->st_ThisComboMetric->GetNumberOfMetrics();

  /** Determine compute range of this thread. */
  const unsigned int subSize = static_cast< unsigned int >(
    vcl_ceil( static_cast< double >( numberOfParameters ) / static_cast< double >( nrOfThreads ) ) );
  unsigned int jmin = threadId * subSize;
  unsigned int jmax = ( threadId + 1 ) * subSize;
  jmax = ( jmax > numberOfParameters ) ? numberOfParameters : jmax;

  // First metric, assign
  if( temp->st_ThisComboMetric->m_UseMetric[ 0 ] )
  {
    double                 weight           = temp->st_ThisComboMetric->GetFinalMetricWeight( 0 );
    const DerivativeType & metricDerivative = temp->st_ThisComboMetric->m_MetricDerivatives[ 0 ];
    for( unsigned int j = jmin; j < jmax; j++ )
    {
      temp->st_Derivative[ j ] = weight * metricDerivative[ j ];
    }
  }

  // Other metrics, add
  for( unsigned int i = 1; i < numberOfMetrics; i++ )
  {
    if( temp->st_ThisComboMetric->m_UseMetric[ i ] )
    {
      double                 weight           = temp->st_ThisComboMetric->GetFinalMetricWeight( i );
      const DerivativeType & metricDerivative = temp->st_ThisComboMetric->m_MetricDerivatives[ i ];
      for( unsigned int j = jmin; j < jmax; j++ )
      {
        temp->st_Derivative[ j ] += weight * metricDerivative[ j ];
      }
    }
  }

  return ITK_THREAD_RETURN_VALUE;

} // end CombineDerivativesThreaderCallback()


/**
 * ********************* GetSelfHessian ****************************
 */

template< class TFixedImage, class TMovingImage >
void
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetSelfHessian( const TransformParametersType & parameters,
  HessianType & H ) const
{
  /** Prepare Hessian */
  H.set_size( this->GetNumberOfParameters(),
    this->GetNumberOfParameters() );
  //H.Fill(0.0);
  HessianType tmpH( this->GetNumberOfParameters(),
  this->GetNumberOfParameters() );

  /** Add all metrics' selfhessians. */
  bool initialized = false;
  for( unsigned int i = 0; i < this->m_NumberOfMetrics; i++ )
  {
    if( this->m_UseMetric[ i ] )
    {
      const double      w      = this->m_MetricWeights[ i ];
      ImageMetricType * metric = dynamic_cast< ImageMetricType * >( this->GetMetric( i ) );
      if( metric )
      {
        initialized = true;
        metric->GetSelfHessian( parameters, tmpH );

        /** H=H+tmpH; \todo: maybe this can be done more efficiently. */
        tmpH.reset();
        while( tmpH.next() )
        {
          H( tmpH.getrow(), tmpH.getcolumn() ) += w * tmpH.value();
        }

      } // end if metric i exists
    }   // end if use metric i
  }     // end for metrics

  /** If none of the submetrics has a valid implementation of GetSelfHessian,
   * then return an identity matrix */
  if( !initialized )
  {
    //H.fill_diagonal(1.0);
    for( unsigned int j = 0; j < this->GetNumberOfParameters(); ++j )
    {
      H( j, j ) = 1.0;
    }
  }

} // end GetSelfHessian()


/**
 * ********************* GetMTime ****************************
 */

template< class TFixedImage, class TMovingImage >
unsigned long
CombinationImageToImageMetric< TFixedImage, TMovingImage >
::GetMTime( void ) const
{
  unsigned long mtime = this->Superclass::GetMTime();
  unsigned long m;

  // Some of the following should be removed once this 'ivars' are put in the
  // input and output lists

  /** Check the modified time of the sub metrics */
  for( unsigned int i = 0; i < this->GetNumberOfMetrics(); ++i )
  {
    SingleValuedCostFunctionPointer metric = this->GetMetric( i );
    if( metric.IsNotNull() )
    {
      m     = metric->GetMTime();
      mtime = ( m > mtime ? m : mtime );
    }
  }

  return mtime;

} // end GetMTime()


} // end namespace itk

#undef itkImplementationSetObjectMacro1
#undef itkImplementationSetObjectMacro2
#undef itkImplementationGetConstObjectMacro1
#undef itkImplementationGetConstObjectMacro2

#endif // end #ifndef _itkCombinationImageToImageMetric_hxx
