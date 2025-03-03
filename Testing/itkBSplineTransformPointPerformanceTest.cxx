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
#include "itkAdvancedBSplineDeformableTransform.h"

#include "itkImageRegionIterator.h"

// Report timings
#include "itkTimeProbe.h"

#include <fstream>
#include <iomanip>

//-------------------------------------------------------------------------------------
// Create a class that inherits from the B-spline transform,
// and adds the previous un-optimized TransformPoint function.
namespace itk
{
template <typename TScalarType = double, unsigned int NDimensions = 3, unsigned int VSplineOrder = 3>
class BSplineTransform_TEST : public AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
{
public:
  /** Standard class typedefs. */
  using Self = BSplineTransform_TEST;
  using Superclass = AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Some stuff that is needed to get this class functional. */
  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(BSplineTransform_TEST);
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::IndexType;
  using typename Superclass::ContinuousIndexType;
  using typename Superclass::WeightsFunctionType;
  using typename Superclass::WeightsType;
  using typename Superclass::ParameterIndexArrayType;
  using typename Superclass::ImageType;
  using typename Superclass::RegionType;
  using typename Superclass::PixelType;
  using typename Superclass::ScalarType;

  /** Transform points by a B-spline deformable transformation. */
  OutputPointType
  TransformPoint_OLD(const InputPointType & point) const
  {
    const unsigned long                         numberOfWeights = WeightsFunctionType::NumberOfWeights;
    typename ParameterIndexArrayType::ValueType indicesArray[numberOfWeights];
    WeightsType                                 weights;
    ParameterIndexArrayType                     indices(indicesArray, numberOfWeights, false);

    OutputPointType outputPoint;
    bool            inside;
    this->TransformPoint_OLD(point, outputPoint, weights, indices, inside);
    return outputPoint;
  } // end TransformPoint_OLD()


  void
  TransformPoint_OLD(const InputPointType &    inputPoint,
                     OutputPointType &         outputPoint,
                     WeightsType &             weights,
                     ParameterIndexArrayType & indices,
                     bool &                    inside) const
  {
    inside = true;
    InputPointType transformedPoint = inputPoint;

    /** Check if the coefficient image has been set. */
    if (!this->m_CoefficientImages[0])
    {
      itkWarningMacro("B-spline coefficients have not been set");
      for (unsigned int j = 0; j < SpaceDimension; ++j)
      {
        outputPoint[j] = transformedPoint[j];
      }
      return;
    }

    /***/
    const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

    // NOTE: if the support region does not lie totally within the grid
    // we assume zero displacement and return the input point
    inside = this->InsideValidRegion(cindex);
    if (!inside)
    {
      outputPoint = transformedPoint;
      return;
    }

    // Compute interpolation weights
    IndexType supportIndex;
    this->m_WeightsFunction->ComputeStartIndex(cindex, supportIndex);
    this->m_WeightsFunction->Evaluate(cindex, supportIndex, weights);

    // For each dimension, correlate coefficient with weights
    RegionType supportRegion;
    supportRegion.SetSize(WeightsFunctionType::SupportSize);
    supportRegion.SetIndex(supportIndex);

    outputPoint.Fill(ScalarType{});

    /** Create iterators over the coefficient images. */
    using IteratorType = ImageRegionConstIterator<ImageType>;
    IteratorType      iterator[SpaceDimension];
    unsigned long     counter = 0;
    const PixelType * basePointer = this->m_CoefficientImages[0]->GetBufferPointer();

    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      iterator[j] = IteratorType(this->m_CoefficientImages[j], supportRegion);
    }

    /** Loop over the support region. */
    while (!iterator[0].IsAtEnd())
    {
      // populate the indices array
      indices[counter] = &(iterator[0].Value()) - basePointer;

      // multiply weigth with coefficient to compute displacement
      for (unsigned int j = 0; j < SpaceDimension; ++j)
      {
        outputPoint[j] += static_cast<ScalarType>(weights[counter] * iterator[j].Value());
        ++iterator[j];
      }
      ++counter;

    } // end while

    // The output point is the start point + displacement.
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      outputPoint[j] += transformedPoint[j];
    }

  } // end TransformPoint_OLD()
};

// end class BSplineTransform_TEST
} // end namespace itk

//-------------------------------------------------------------------------------------

int
main(int argc, char * argv[])
{
  /** Some basic type definitions.
   * NOTE: don't change the dimension or the spline order, since the
   * hard-coded ground truth depends on this.
   */
  const unsigned int Dimension = 3;
  const unsigned int SplineOrder = 3;
  using CoordinateRepresentationType = double;

  /** The number of calls to Evaluate(). Distinguish between
   * Debug and Release mode.
   */
#ifndef NDEBUG
  auto N = static_cast<unsigned int>(1e3);
#else
  unsigned int N = static_cast<unsigned int>(1e5);
#endif
  std::cerr << "N = " << N << std::endl;

  /** Check. */
  if (argc != 2)
  {
    std::cerr << "ERROR: You should specify a text file with the B-spline transformation parameters." << std::endl;
    return 1;
  }

  /** Typedefs. */
  using TransformType = itk::BSplineTransform_TEST<CoordinateRepresentationType, Dimension, SplineOrder>;

  using InputPointType = TransformType::InputPointType;
  using OutputPointType = TransformType::OutputPointType;
  using ParametersType = TransformType::ParametersType;

  using InputImageType = itk::Image<CoordinateRepresentationType, Dimension>;
  using RegionType = InputImageType::RegionType;
  using SizeType = InputImageType::SizeType;
  using IndexType = InputImageType::IndexType;
  using SpacingType = InputImageType::SpacingType;
  using OriginType = InputImageType::PointType;
  using DirectionType = InputImageType::DirectionType;

  /** Create the transform. */
  auto transform = TransformType::New();

  /** Setup the B-spline transform:
   * (GridSize 44 43 35)
   * (GridIndex 0 0 0)
   * (GridSpacing 10.7832773148 11.2116431394 11.8648235177)
   * (GridOrigin -237.6759555555 -239.9488431747 -344.2315805162)
   */
  SizeType gridSize;
  gridSize[0] = 44;
  gridSize[1] = 43;
  gridSize[2] = 35;
  IndexType  gridIndex{};
  RegionType gridRegion;
  gridRegion.SetSize(gridSize);
  gridRegion.SetIndex(gridIndex);
  SpacingType gridSpacing;
  gridSpacing[0] = 10.7832773148;
  gridSpacing[1] = 11.2116431394;
  gridSpacing[2] = 11.8648235177;
  OriginType gridOrigin;
  gridOrigin[0] = -237.6759555555;
  gridOrigin[1] = -239.9488431747;
  gridOrigin[2] = -344.2315805162;

  transform->SetGridOrigin(gridOrigin);
  transform->SetGridSpacing(gridSpacing);
  transform->SetGridRegion(gridRegion);
  transform->SetGridDirection(DirectionType::GetIdentity());

  /** Now read the parameters as defined in the file par.txt. */
  ParametersType parameters(transform->GetNumberOfParameters());
  std::ifstream  input(argv[1]);
  if (input.is_open())
  {
    for (unsigned int i = 0; i < parameters.GetSize(); ++i)
    {
      input >> parameters[i];
    }
  }
  else
  {
    std::cerr << "ERROR: could not open the text file containing the parameter values." << std::endl;
    return 1;
  }
  transform->SetParameters(parameters);

  /** Declare variables. */
  auto            inputPoint = itk::MakeFilled<InputPointType>(4.1);
  OutputPointType outputPoint;
  double          sum = 0.0;
  itk::TimeProbe  timeProbeOLD, timeProbeNEW;

  /** Time the TransformPoint with the old region iterator. */
  timeProbeOLD.Start();
  for (unsigned int i = 0; i < N; ++i)
  {
    outputPoint = transform->TransformPoint_OLD(inputPoint);
    sum += outputPoint[0];
    sum += outputPoint[1];
    sum += outputPoint[2];
  }
  timeProbeOLD.Stop();
  const double oldTime = timeProbeOLD.GetMean();

  /** Time the TransformPoint with the new scanline iterator. */
  timeProbeNEW.Start();
  for (unsigned int i = 0; i < N; ++i)
  {
    outputPoint = transform->TransformPoint(inputPoint);
    sum += outputPoint[0];
    sum += outputPoint[1];
    sum += outputPoint[2];
  }
  timeProbeNEW.Stop();
  const double newTime = timeProbeNEW.GetMean();

  // Avoid compiler optimizations, so use sum
  std::cerr << sum << std::endl; // works but ugly on screen
  //  volatile double a = sum; // works but gives unused variable warning
  // #pragma optimize( "", off ) // unrecognized pragma
  // sum += 2.0;

  /** Report timings. */
  std::cerr << std::setprecision(4);
  std::cerr << "Time OLD = " << oldTime << " " << timeProbeOLD.GetUnit() << std::endl;
  std::cerr << "Time NEW = " << newTime << " " << timeProbeNEW.GetUnit() << std::endl;
  std::cerr << "Speedup factor = " << oldTime / newTime << std::endl;

  /** Return a value. */
  return 0;

} // end main
