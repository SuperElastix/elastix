/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkFullSearchOptimizer_h
#define __itkFullSearchOptimizer_h

#include "itkSingleValuedNonLinearOptimizer.h"
#include "itkMapContainer.h"
#include "itkImage.h"
#include "itkArray.h"
#include "itkFixedArray.h"

namespace itk
{

/**
 * \class FullSearchOptimizer
 * \brief An optimizer based on full search.
 *
 * Optimizer that scans a subspace of the parameter space
 * and searches for the best parameters.
 *
 * \todo This optimizer has similar functionality as the recently added
 * itkExhaustiveOptimizer. See if we can replace it by that optimizer,
 * or inherit from it.
 *
 * \ingroup Optimizers
 * \sa FullSearch
 */

class FullSearchOptimizer : public SingleValuedNonLinearOptimizer
{
public:

  /** Standard class typedefs. */
  typedef FullSearchOptimizer            Self;
  typedef SingleValuedNonLinearOptimizer Superclass;
  typedef SmartPointer< Self >           Pointer;
  typedef SmartPointer< const Self >     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( FullSearchOptimizer, SingleValuedNonLinearOptimizer );

  /** Codes of stopping conditions */
  typedef enum {
    FullRangeSearched,
    MetricError
  } StopConditionType;

  /* Typedefs inherited from superclass */
  typedef Superclass::ParametersType      ParametersType;
  typedef Superclass::CostFunctionType    CostFunctionType;
  typedef Superclass::CostFunctionPointer CostFunctionPointer;
  typedef Superclass::MeasureType         MeasureType;

  typedef ParametersType::ValueType               ParameterValueType;     // = double
  typedef ParameterValueType                      RangeValueType;
  typedef FixedArray< RangeValueType, 3 >         RangeType;
  typedef MapContainer< unsigned int, RangeType > SearchSpaceType;
  typedef SearchSpaceType::Pointer                SearchSpacePointer;
  typedef SearchSpaceType::ConstIterator          SearchSpaceIteratorType;

  /** Type that stores the parameter values of the parameters to be optimized.
  * Updated every iteration. */
  typedef Array< ParameterValueType > SearchSpacePointType;

  /** The same values, but transformed to integer indices.
  * These can be used to create an image visualizing the search space. */
  typedef Array< IndexValueType > SearchSpaceIndexType;

  /** The size of each dimension to be searched ((max-min)/step)) */
  typedef Array< SizeValueType > SearchSpaceSizeType;

  /** NB: The methods SetScales has no influence! */

  /** Methods to configure the cost function. */
  itkGetConstMacro( Maximize, bool );
  itkSetMacro( Maximize, bool );
  itkBooleanMacro( Maximize );
  bool GetMinimize() const
  { return !m_Maximize; }
  void SetMinimize( bool v )
  { this->SetMaximize( !v ); }
  void MinimizeOn()
  { this->MaximizeOff(); }
  void MinimizeOff()
  { this->MaximizeOn(); }

  /** Set the CurrentPosition, CurrentPoint and CurrentIndex to the next point
   * in the search space.
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
  virtual void UpdateCurrentPosition( void );

  /** Start optimization.
   * Make sure to set the initial position before starting the optimization
   */
  virtual void StartOptimization( void );

  /** Resume previously stopped optimization with current parameters
   * \sa StopOptimization.
   */
  virtual void ResumeOptimization( void );

  /** Stop optimization.
   * \sa ResumeOptimization
   */
  virtual void StopOptimization( void );

  /**
   * Set/Get the SearchSpace, which is defined by a pointer to an
   * itkMapContainer<unsigned int, FixedArray(double,3)>
   * The unsigned int is the number of a parameter to be
   * investigated. The FixedArray contains its range and the
   * resolution of the search (min, max, step).
   *
   * Instead of using this function, the Add/RemoveSearchDimension methods can be used,
   * to define a search space.
   */
  itkSetObjectMacro( SearchSpace, SearchSpaceType );
  itkGetObjectMacro( SearchSpace, SearchSpaceType );

  /** Add/Remove a dimension to/from the SearchSpace */
  virtual void AddSearchDimension( unsigned int param_nr,
    RangeValueType minimum, RangeValueType maximum, RangeValueType step );

  virtual void RemoveSearchDimension( unsigned int param_nr );

  /** Get the total number of iterations = sizes[0]*sizes[1]*sizes[2]* etc..... */
  virtual unsigned long GetNumberOfIterations( void );

  /** Get the Dimension of the SearchSpace. Calculated from the SearchSpace. */
  virtual unsigned int GetNumberOfSearchSpaceDimensions( void );

  /** Returns an array containing trunc((max-min)/step) for each SearchSpaceDimension) */
  virtual const SearchSpaceSizeType & GetSearchSpaceSize( void );

  /** Convert an index to a full parameter array. Requires a valid InitialPosition! */
  virtual ParametersType PointToPosition( const SearchSpacePointType & point );

  virtual ParametersType IndexToPosition( const SearchSpaceIndexType & index );

  /** Convert an index to a point */
  virtual SearchSpacePointType IndexToPoint( const SearchSpaceIndexType & index );

  /** Get the current iteration number. */
  itkGetConstMacro( CurrentIteration, unsigned long );

  /** Get the point in SearchSpace that is currently evaluated */
  itkGetConstReferenceMacro( CurrentPointInSearchSpace, SearchSpacePointType );
  itkGetConstReferenceMacro( CurrentIndexInSearchSpace, SearchSpaceIndexType );

  /** Get the point in SearchSpace that is currently the most optimal */
  itkGetConstReferenceMacro( BestPointInSearchSpace, SearchSpacePointType );
  itkGetConstReferenceMacro( BestIndexInSearchSpace, SearchSpaceIndexType );

  /** Get the current value. */
  itkGetConstMacro( Value, double );

  /** Get the best value. */
  itkGetConstMacro( BestValue, double );

  /** Get Stop condition. */
  itkGetConstMacro( StopCondition, StopConditionType );

protected:

  FullSearchOptimizer();
  virtual ~FullSearchOptimizer() {}

  //void PrintSelf(std::ostream& os, Indent indent) const;

  bool              m_Maximize;
  bool              m_Stop;
  double            m_Value;
  double            m_BestValue;
  StopConditionType m_StopCondition;

  SearchSpacePointer   m_SearchSpace;
  SearchSpacePointType m_CurrentPointInSearchSpace;
  SearchSpaceIndexType m_CurrentIndexInSearchSpace;
  SearchSpacePointType m_BestPointInSearchSpace;
  SearchSpaceIndexType m_BestIndexInSearchSpace;
  SearchSpaceSizeType  m_SearchSpaceSize;
  unsigned int         m_NumberOfSearchSpaceDimensions;

  unsigned long m_LastSearchSpaceChanges;
  virtual void ProcessSearchSpaceChanges( void );

private:

  FullSearchOptimizer( const Self & ); // purposely not implemented
  void operator=( const Self & );      // purposely not implemented

  unsigned long m_CurrentIteration;

};

} // end namespace itk

#endif
