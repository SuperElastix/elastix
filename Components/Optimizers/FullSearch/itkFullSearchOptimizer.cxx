/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __itkFullSearchOptimizer_cxx
#define __itkFullSearchOptimizer_cxx

#include "itkFullSearchOptimizer.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkExceptionObject.h"
#include "itkNumericTraits.h"

namespace itk
{

/**
 * ************************ Constructor **************************
 */
FullSearchOptimizer
::FullSearchOptimizer()
{
  itkDebugMacro( "Constructor" );

  m_CurrentIteration              = 0;
  m_Maximize                      = false;
  m_Value                         = 0.0;
  m_BestValue                     = 0.0;
  m_StopCondition                 = FullRangeSearched;
  m_Stop                          = false;
  m_NumberOfSearchSpaceDimensions = 0;
  m_SearchSpace                   = 0;
  m_LastSearchSpaceChanges        = 0;

}   //end constructor


/**
 * ***************** Start the optimization **********************
 */
void
FullSearchOptimizer
::StartOptimization( void )
{

  itkDebugMacro( "StartOptimization" );

  m_CurrentIteration = 0;

  this->ProcessSearchSpaceChanges();

  m_CurrentIndexInSearchSpace.Fill( 0 );
  m_BestIndexInSearchSpace.Fill( 0 );

  m_CurrentPointInSearchSpace = this->IndexToPoint( m_CurrentIndexInSearchSpace );
  m_BestPointInSearchSpace    = m_CurrentPointInSearchSpace;

  this->SetCurrentPosition( this->PointToPosition( m_CurrentPointInSearchSpace ) );

  if( m_Maximize )
  {
    m_BestValue = NumericTraits< double >::NonpositiveMin();
  }
  else
  {
    m_BestValue = NumericTraits< double >::max();
  }

  this->ResumeOptimization();

}


/**
 * ******************** Resume the optimization ******************
 */
void
FullSearchOptimizer
::ResumeOptimization( void )
{

  itkDebugMacro( "ResumeOptimization" );

  m_Stop = false;

  InvokeEvent( StartEvent() );
  while( !m_Stop )
  {

    try
    {
      m_Value = m_CostFunction->GetValue( this->GetCurrentPosition() );
    }
    catch( ExceptionObject & err )
    {
      // An exception has occurred.
      // Terminate immediately.
      m_StopCondition = MetricError;
      StopOptimization();

      // Pass exception to caller
      throw err;
    }

    if( m_Stop )
    {
      break;
    }

    /** Check if the value is a minimum or maximum */
    if( ( m_Value < m_BestValue )  ^  m_Maximize )         // ^ = xor, yields true if only one of the expressions is true
    {
      m_BestValue              = m_Value;
      m_BestPointInSearchSpace = m_CurrentPointInSearchSpace;
      m_BestIndexInSearchSpace = m_CurrentIndexInSearchSpace;
    }

    this->InvokeEvent( IterationEvent() );

    /** Prepare for next step */
    m_CurrentIteration++;

    if( m_CurrentIteration >= this->GetNumberOfIterations() )
    {
      m_StopCondition = FullRangeSearched;
      StopOptimization();
      break;
    }

    /** Set the next position in search space. */
    this->UpdateCurrentPosition();

  }   // end while

}   //end function ResumeOptimization


/**
 * ************************** Stop optimization ******************
 */
void
FullSearchOptimizer
::StopOptimization( void )
{

  itkDebugMacro( "StopOptimization" );

  m_Stop = true;

  this->SetCurrentPosition(
    this->PointToPosition( m_BestPointInSearchSpace ) );
  InvokeEvent( EndEvent() );

}   // end function StopOptimization


/**
 * ********************* UpdateCurrentPosition *******************
 *
 * Goes to the next point in search space
 *
 * example of sequence of indices in a 3d search space:
 *
 * dim1: 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2
 * dim2: 0 0 0 1 1 1 2 2 2 0 0 0 1 1 1 2 2 2 0 0 0 1 1 1 2 2 2
 * dim3: 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2
 *
 * The indices are transformed to points in search space with the formula:
 * point[i] = min[i] + stepsize[i]*index[i]       for all i.
 *
 * Then the appropriate parameters in the ParameterArray are updated.
 */

void
FullSearchOptimizer
::UpdateCurrentPosition( void )
{

  itkDebugMacro( "Current position updated." );

  /** Get the current parameters; const_cast, because we want to adapt it later. */
  ParametersType & currentPosition = const_cast< ParametersType & >( this->GetCurrentPosition() );

  /** Get the dimension and sizes of the searchspace. */
  const unsigned int          searchSpaceDimension = this->GetNumberOfSearchSpaceDimensions();
  const SearchSpaceSizeType & searchSpaceSize      = this->GetSearchSpaceSize();

  /** Derive the index of the next search space point */
  bool JustSetPreviousDimToZero = true;
  for( unsigned int ssdim = 0; ssdim < searchSpaceDimension; ssdim++ )  //loop over all dimensions of the search space
  {
    /** if the full range of ssdim-1 has been searched (so, if its
     * index has just been set back to 0) then increase index[ssdim] */
    if( JustSetPreviousDimToZero )
    {
      /** reset the bool */
      JustSetPreviousDimToZero = false;

      /** determine the new value of m_CurrentIndexInSearchSpace[ssdim] */
      unsigned int dummy = m_CurrentIndexInSearchSpace[ ssdim ] + 1;
      if( dummy == searchSpaceSize[ ssdim ] )
      {
        m_CurrentIndexInSearchSpace[ ssdim ] = 0;
        JustSetPreviousDimToZero             = true;
      }
      else
      {
        m_CurrentIndexInSearchSpace[ ssdim ] = dummy;
      }
    }   // end if justsetprevdimtozero

  }   // end for

  /** Initialise the iterator. */
  SearchSpaceIteratorType it( m_SearchSpace->Begin() );

  /** Transform the index to a point in search space.
   * Change the appropriate parameters in the ParameterArray.
   *
   * The IndexToPoint and PointToParameter functions are not used here,
   * because we edit directly in the currentPosition (faster).
   */
  for( unsigned int ssdim = 0; ssdim < searchSpaceDimension; ssdim++ )
  {
    /** Transform the index to a point; point = min + step*index */
    RangeType range = it.Value();
    m_CurrentPointInSearchSpace[ ssdim ] = range[ 0 ]
      + static_cast< double >( range[ 2 ] * m_CurrentIndexInSearchSpace[ ssdim ] );

    /** Update the array of parameters. */
    currentPosition[ it.Index() ] = m_CurrentPointInSearchSpace[ ssdim ];
    it++;
  }   // end for

}   // end UpdateCurrentPosition


/**
 * ********************* ProcessSearchSpaceChanges **************
 */
void
FullSearchOptimizer
::ProcessSearchSpaceChanges( void )
{
  if( m_SearchSpace->GetMTime() > m_LastSearchSpaceChanges )
  {

    /** Update the number of search space dimensions. */
    m_NumberOfSearchSpaceDimensions = static_cast< unsigned int >( m_SearchSpace->Size() );

    /** Set size of arrays accordingly */
    m_SearchSpaceSize.SetSize( m_NumberOfSearchSpaceDimensions );
    m_CurrentIndexInSearchSpace.SetSize( m_NumberOfSearchSpaceDimensions );
    m_CurrentPointInSearchSpace.SetSize( m_NumberOfSearchSpaceDimensions );
    m_BestIndexInSearchSpace.SetSize( m_NumberOfSearchSpaceDimensions );
    m_BestPointInSearchSpace.SetSize( m_NumberOfSearchSpaceDimensions );

    /** Initialise an iterator over the search space map. */
    SearchSpaceIteratorType it( m_SearchSpace->Begin() );

    for( unsigned int ssdim = 0; ssdim < m_NumberOfSearchSpaceDimensions; ssdim++ )
    {
      RangeType range = it.Value();
      m_SearchSpaceSize[ ssdim ] = static_cast< unsigned long >( ( range[ 1 ] - range[ 0 ] ) / range[ 2 ] ) + 1;
      it++;
    }

  }   // end if search space modified

  /** Remember the time of the last processed changes */
  m_LastSearchSpaceChanges = m_SearchSpace->GetMTime();

}   // end function ProcessSearchSpaceChanges


/**
 * ********************** AddSearchDimension ********************
 *
 * Add a dimension to the SearchSpace
 */
void
FullSearchOptimizer
::AddSearchDimension(
  unsigned int param_nr,
  RangeValueType minimum,
  RangeValueType maximum,
  RangeValueType step )
{
  if( !m_SearchSpace )
  {
    m_SearchSpace = SearchSpaceType::New();
  }

  /** Fill a range array */
  RangeType range;
  range[ 0 ] = minimum;
  range[ 1 ] = maximum;
  range[ 2 ] = step;

  /** Delete the range if it already was defined before */
  m_SearchSpace->DeleteIndex( param_nr );

  /** Insert the new range specification */
  m_SearchSpace->InsertElement( param_nr, range );
}


/**
 * ******************* RemoveSearchDimension ********************
 *
 * Remove a dimension from the SearchSpace
 */
void
FullSearchOptimizer
::RemoveSearchDimension( unsigned int param_nr )
{
  if( m_SearchSpace )
  {
    m_SearchSpace->DeleteIndex( param_nr );
  }
}


/**
 * ***************** GetNumberOfIterations **********************
 *
 * Get the total number of iterations = sizes[0]*sizes[1]*sizes[2]* etc.....
 */
unsigned long
FullSearchOptimizer
::GetNumberOfIterations( void )
{
  SearchSpaceSizeType sssize   = this->GetSearchSpaceSize();
  unsigned int        maxssdim = this->GetNumberOfSearchSpaceDimensions();
  unsigned long       nr_it    = 0;

  if( maxssdim > 0 )
  {
    nr_it = sssize[ 0 ];
    for( unsigned int ssdim = 1; ssdim < maxssdim; ssdim++ )
    {
      nr_it *= sssize[ ssdim ];
    }
  }   // end if

  return nr_it;
}


/**
 * ******************** GetNumberOfSearchSpaceDimensions ********
 *
 * Get the Dimension of the SearchSpace.
 */
unsigned int
FullSearchOptimizer
::GetNumberOfSearchSpaceDimensions( void )
{
  this->ProcessSearchSpaceChanges();
  return this->m_NumberOfSearchSpaceDimensions;
}


/**
 * ******************** GetSearchSpaceSize **********************
 *
 * Returns an array containing trunc((max-min)/step) for each
 * SearchSpaceDimension)
 */
const FullSearchOptimizer::SearchSpaceSizeType &
FullSearchOptimizer
::GetSearchSpaceSize( void )
{
  this->ProcessSearchSpaceChanges();
  return this->m_SearchSpaceSize;
}


/**
 * ********************* PointToPosition ************************
 */
FullSearchOptimizer::ParametersType
FullSearchOptimizer
::PointToPosition( const SearchSpacePointType & point )
{
  const unsigned int searchSpaceDimension = this->GetNumberOfSearchSpaceDimensions();

  /** \todo check if point has the same dimension. */

  ParametersType param = this->GetInitialPosition();

  /** Initialise the iterator. */
  SearchSpaceIteratorType it( m_SearchSpace->Begin() );

  /** Transform the index to a point in search space. */
  for( unsigned int ssdim = 0; ssdim < searchSpaceDimension; ssdim++ )
  {
    /** Update the array of parameters. */
    param[ it.Index() ] = point[ ssdim ];

    /** go to next dimension in search space */
    it++;

  }

  return param;

}   // end point to position


/**
 * ********************* IndexToPosition ************************
 */
FullSearchOptimizer::ParametersType
FullSearchOptimizer
::IndexToPosition( const SearchSpaceIndexType & index )
{
  return this->PointToPosition( this->IndexToPoint( index ) );
}


/**
 * ********************* IndexToPoint ***************************
 */
FullSearchOptimizer::SearchSpacePointType
FullSearchOptimizer
::IndexToPoint( const SearchSpaceIndexType & index )
{

  const unsigned int   searchSpaceDimension = this->GetNumberOfSearchSpaceDimensions();
  SearchSpacePointType point( searchSpaceDimension );

  /** Initialise the iterator. */
  SearchSpaceIteratorType it( m_SearchSpace->Begin() );

  /** Transform the index to a point in search space. */
  for( unsigned int ssdim = 0; ssdim < searchSpaceDimension; ssdim++ )
  {
    /** point = min + step*index */
    RangeType range = it.Value();
    point[ ssdim ] = range[ 0 ] + static_cast< double >( range[ 2 ] * index[ ssdim ] );

    /** go to next dimension in search space */
    it++;
  }   // end for

  return point;

}   // end IndexToPoint


} // end namespace itk

#endif // #ifndef __itkFullSearchOptimizer_cxx
