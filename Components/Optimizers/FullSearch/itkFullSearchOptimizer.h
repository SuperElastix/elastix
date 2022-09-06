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

#ifndef itkFullSearchOptimizer_h
#define itkFullSearchOptimizer_h

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
  ITK_DISALLOW_COPY_AND_MOVE(FullSearchOptimizer);

  /** Standard class typedefs. */
  using Self = FullSearchOptimizer;
  using Superclass = SingleValuedNonLinearOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FullSearchOptimizer, SingleValuedNonLinearOptimizer);

  /** Codes of stopping conditions */
  enum StopConditionType
  {
    FullRangeSearched,
    MetricError
  };

  /* Typedefs inherited from superclass */
  using Superclass::ParametersType;
  using Superclass::CostFunctionType;
  using Superclass::CostFunctionPointer;
  using Superclass::MeasureType;

  using ParameterValueType = ParametersType::ValueType; // = double
  using RangeValueType = ParameterValueType;
  using RangeType = FixedArray<RangeValueType, 3>;
  using SearchSpaceType = MapContainer<unsigned int, RangeType>;
  using SearchSpacePointer = SearchSpaceType::Pointer;
  using SearchSpaceIteratorType = SearchSpaceType::ConstIterator;

  /** Type that stores the parameter values of the parameters to be optimized.
   * Updated every iteration. */
  using SearchSpacePointType = Array<ParameterValueType>;

  /** The same values, but transformed to integer indices.
   * These can be used to create an image visualizing the search space. */
  using SearchSpaceIndexType = Array<IndexValueType>;

  /** The size of each dimension to be searched ((max-min)/step)) */
  using SearchSpaceSizeType = Array<SizeValueType>;

  /** NB: The methods SetScales has no influence! */

  /** Methods to configure the cost function. */
  itkGetConstMacro(Maximize, bool);
  itkSetMacro(Maximize, bool);
  itkBooleanMacro(Maximize);
  bool
  GetMinimize() const
  {
    return !m_Maximize;
  }
  void
  SetMinimize(bool v)
  {
    this->SetMaximize(!v);
  }
  void
  MinimizeOn()
  {
    this->MaximizeOff();
  }
  void
  MinimizeOff()
  {
    this->MaximizeOn();
  }

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
  virtual void
  UpdateCurrentPosition();

  /** Start optimization.
   * Make sure to set the initial position before starting the optimization
   */
  void
  StartOptimization() override;

  /** Resume previously stopped optimization with current parameters
   * \sa StopOptimization.
   */
  virtual void
  ResumeOptimization();

  /** Stop optimization.
   * \sa ResumeOptimization
   */
  virtual void
  StopOptimization();

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
  itkSetObjectMacro(SearchSpace, SearchSpaceType);
  itkGetModifiableObjectMacro(SearchSpace, SearchSpaceType);

  /** Add/Remove a dimension to/from the SearchSpace */
  virtual void
  AddSearchDimension(unsigned int param_nr, RangeValueType minimum, RangeValueType maximum, RangeValueType step);

  virtual void
  RemoveSearchDimension(unsigned int param_nr);

  /** Get the total number of iterations = sizes[0]*sizes[1]*sizes[2]* etc..... */
  virtual unsigned long
  GetNumberOfIterations();

  /** Get the Dimension of the SearchSpace. Calculated from the SearchSpace. */
  virtual unsigned int
  GetNumberOfSearchSpaceDimensions();

  /** Returns an array containing trunc((max-min)/step) for each SearchSpaceDimension) */
  virtual const SearchSpaceSizeType &
  GetSearchSpaceSize();

  /** Convert an index to a full parameter array. Requires a valid InitialPosition! */
  virtual ParametersType
  PointToPosition(const SearchSpacePointType & point);

  virtual ParametersType
  IndexToPosition(const SearchSpaceIndexType & index);

  /** Convert an index to a point */
  virtual SearchSpacePointType
  IndexToPoint(const SearchSpaceIndexType & index);

  /** Get the current iteration number. */
  itkGetConstMacro(CurrentIteration, unsigned long);

  /** Get the point in SearchSpace that is currently evaluated */
  itkGetConstReferenceMacro(CurrentPointInSearchSpace, SearchSpacePointType);
  itkGetConstReferenceMacro(CurrentIndexInSearchSpace, SearchSpaceIndexType);

  /** Get the point in SearchSpace that is currently the most optimal */
  itkGetConstReferenceMacro(BestPointInSearchSpace, SearchSpacePointType);
  itkGetConstReferenceMacro(BestIndexInSearchSpace, SearchSpaceIndexType);

  /** Get the current value. */
  itkGetConstMacro(Value, double);

  /** Get the best value. */
  itkGetConstMacro(BestValue, double);

  /** Get Stop condition. */
  itkGetConstMacro(StopCondition, StopConditionType);

protected:
  FullSearchOptimizer();
  ~FullSearchOptimizer() override = default;

  // void PrintSelf(std::ostream& os, Indent indent) const;

  bool              m_Maximize{ false };
  bool              m_Stop{ false };
  double            m_Value{ 0.0 };
  double            m_BestValue{ 0.0 };
  StopConditionType m_StopCondition{ FullRangeSearched };

  SearchSpacePointer   m_SearchSpace{ nullptr };
  SearchSpacePointType m_CurrentPointInSearchSpace;
  SearchSpaceIndexType m_CurrentIndexInSearchSpace;
  SearchSpacePointType m_BestPointInSearchSpace;
  SearchSpaceIndexType m_BestIndexInSearchSpace;
  SearchSpaceSizeType  m_SearchSpaceSize;
  unsigned int         m_NumberOfSearchSpaceDimensions{ 0 };

  unsigned long m_LastSearchSpaceChanges{ 0 };
  virtual void
  ProcessSearchSpaceChanges();

private:
  unsigned long m_CurrentIteration{ 0 };
};

} // end namespace itk

#endif
