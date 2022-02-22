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
#include "itkBSplineInterpolationSecondOrderDerivativeWeightFunction.h"

#include <array>
#include <ctime>
#include <iomanip>

//-------------------------------------------------------------------------------------

int
main()
{
  /** Some basic type definitions.
   * NOTE: don't change the dimension or the spline order, since the
   * hard-coded ground truth depends on this.
   */
  const unsigned int Dimension = 2;
  const unsigned int SplineOrder = 3;
  using CoordinateRepresentationType = float;
  const double distance = 1e-3; // the allowable distance
  // const double allowedTimeDifference = 0.1; // 10% is considered within limits
  /** The number of calls to Evaluate(). This number gives reasonably
   * fast test results in Release mode.
   */
  unsigned int N = static_cast<unsigned int>(1e6);

  /** Other typedefs. */
  using SODerivativeWeightFunctionType =
    itk::BSplineInterpolationSecondOrderDerivativeWeightFunction<CoordinateRepresentationType, Dimension, SplineOrder>;
  using ContinuousIndexType = SODerivativeWeightFunctionType::ContinuousIndexType;
  using WeightsType = SODerivativeWeightFunctionType::WeightsType;

  std::cerr << "TESTING:\n" << std::endl;

  /**
   * *********** TESTING 1 ************************************************
   */

  std::cerr << "\nTESTING: derivatives (0,1)\n" << std::endl;

  /** Construct several weight functions. */
  auto soWeightFunction = SODerivativeWeightFunctionType::New();

  /** Create and fill a continuous index.
   * NOTE: don't change this, since the hard-coded ground truth depends on this.
   */
  ContinuousIndexType cindex;
  cindex[0] = 3.1f;
  cindex[1] = -2.2f;
  soWeightFunction->SetDerivativeDirections(0, 1);

  /** Run evaluate for the second order derivative. */
  WeightsType soWeights = soWeightFunction->Evaluate(cindex);
  std::cerr << "weights (2nd order) " << soWeights << std::endl;

  /** Hard code the ground truth. You should change this if you change the
   * spline order.
   *
   * x1 =  3.1  ->  support y1 =  2  3  4  5  ->  x1 - y1 = 1.1 0.1 -0.9 -1.9
   * x2 = -2.2  ->  support y2 = -4 -3 -3 -1  ->  x2 - y2 = 1.8 0.8 -0.2 -1.2
   *
   * B3 is the third order B-spline. ?etc means repeat ? for ever.
   * The coefficients are:
   *   [ B2(x1-y1i+1/2)-B2(x1-y1i-1/2) ] * [ B2(x2-y2i+1/2)-B2(x2-y2i-1/2) ]
   *
   * B3d(  1.1 ) = -0.405
   * B3d(  0.1 ) = -0.185
   * B3d( -0.9 ) =  0.585
   * B3d( -1.9 ) =  0.005
   * B3d(  1.8 ) = -0.02
   * B3d(  0.8 ) = -0.64
   * B3d( -0.2 ) =  0.34
   * B3d( -1.2 ) =  0.32
   *
   *                       -> i
   *       0.0081    0.0037   -0.0117   -0.0001
   *  |    0.2592    0.1184   -0.3744   -0.0032
   *  j   -0.1377   -0.0629    0.1989    0.0017
   *      -0.1296   -0.0592    0.1872    0.0016
   *
   * These numbers are created by a small Matlab program. So, if this appears
   * to be not a valid check, then we made the same bug twice.
   */
  std::array<double, WeightsType::Dimension> trueSOWeights = { { 0.0081,
                                                                 0.0037,
                                                                 -0.0117,
                                                                 -0.0001,
                                                                 0.2592,
                                                                 0.1184,
                                                                 -0.3744,
                                                                 -0.0032,
                                                                 -0.1377,
                                                                 -0.0629,
                                                                 0.1989,
                                                                 0.0017,
                                                                 -0.1296,
                                                                 -0.0592,
                                                                 0.1872,
                                                                 0.0016 } };

  /** Compute the distance between the two vectors. */
  double error = 0.0;
  for (unsigned int i = 0; i < soWeights.Size(); ++i)
  {
    error += vnl_math::sqr(soWeights[i] - trueSOWeights[i]);
  }
  error = std::sqrt(error);

  /** TEST: Compare the two qualitatively. */
  if (error > distance)
  {
    std::cerr << "ERROR: the first order weights differs more than " << distance << " from the truth." << std::endl;
    return 1;
  }
  std::cerr << std::showpoint;
  std::cerr << std::scientific;
  std::cerr << std::setprecision(4);
  std::cerr << "The distance is: " << error << std::endl;

  /** Time the so implementation. */
  clock_t startClock = clock();
  for (unsigned int i = 0; i < N; ++i)
  {
    soWeightFunction->Evaluate(cindex);
  }
  clock_t endClock = clock();
  clock_t elapsed = endClock - startClock;
  std::cerr << "The elapsed time for the 2nd order derivative (0,1) is: " << elapsed << std::endl;

  /**
   * *********** TESTING 2 ************************************************
   */

  std::cerr << "\n--------------------------------------------------------";
  std::cerr << "\nTESTING: derivatives (0,0)\n" << std::endl;

  /** Run evaluate for the second order derivative. */
  soWeightFunction->SetDerivativeDirections(0, 0);
  soWeights = soWeightFunction->Evaluate(cindex);
  std::cerr << "weights (2nd order) " << soWeights << std::endl;

  /** Hard code the ground truth. You should change this if you change the
   * spline order.
   *
   * x1 =  3.1  ->  support y1 =  2  3  4  5  ->  x1 - y1 = 1.1 0.1 -0.9 -1.9
   * x2 = -2.2  ->  support y2 = -4 -3 -3 -1  ->  x2 - y2 = 1.8 0.8 -0.2 -1.2
   *
   * B3 is the third order B-spline. ?etc means repeat ? for ever.
   * The coefficients are:
   *   [ B1(x1-y1i+1) - 2*B1(x1-y1i) + B1(x1-y1i-1) ] * B3(x2-y2i)
   *
   * B3dd(  1.1 ) =  0.9
   * B3dd(  0.1 ) = -1.7
   * B3dd( -0.9 ) =  0.7
   * B3dd( -1.9 ) =  0.1
   * B3  (  1.8 ) =  0.0013etc
   * B3  (  0.8 ) =  0.2826etc
   * B3  ( -0.2 ) =  0.6306etc
   * B3  ( -1.2 ) =  0.0853etc
   *
   *                       -> i
   *      0.0012  -0.00226/   9.3/e-4    1.3/e-4
   *  |   0.2544  -0.48053/   0.19786/   0.02826/
   *  j   0.5676  -1.07213/   0.44146/   0.06306/
   *      0.0768  -0.14506/   0.05973/   0.00853/
   *
   * These numbers are created by a small Matlab program. So, if this appears
   * to be not a valid check, then we made the same bug twice.
   */
  trueSOWeights = { { 1.200000000000e-3,
                      -2.266666666666e-3,
                      9.333333333333e-4,
                      1.333333333333e-4,
                      2.544000000000e-1,
                      -4.805333333333e-1,
                      1.978666666666e-1,
                      2.826666666666e-2,
                      5.676000000000e-1,
                      -1.072133333333,
                      4.414666666666e-1,
                      6.306666666666e-2,
                      7.680000000000e-2,
                      -1.450666666666e-1,
                      5.973333333333e-2,
                      8.533333333333e-3 } };

  /** Compute the distance between the two vectors. */
  error = 0.0;
  for (unsigned int i = 0; i < soWeights.Size(); ++i)
  {
    error += vnl_math::sqr(soWeights[i] - trueSOWeights[i]);
  }
  error = std::sqrt(error);

  /** TEST: Compare the two qualitatively. */
  if (error > distance)
  {
    std::cerr << "ERROR: the first order weights differs more than " << distance << " from the truth." << std::endl;
    return 1;
  }
  std::cerr << std::showpoint;
  std::cerr << std::scientific;
  std::cerr << std::setprecision(4);
  std::cerr << "The distance is: " << error << std::endl;

  /** Time the so implementation. */
  startClock = clock();
  for (unsigned int i = 0; i < N; ++i)
  {
    soWeightFunction->Evaluate(cindex);
  }
  endClock = clock();
  elapsed = endClock - startClock;
  std::cerr << "The elapsed time for the 2nd order derivative (0,0) is: " << elapsed << std::endl;

  /**
   * *********** Function TESTING ****************************************
   */

  std::cerr << "\n--------------------------------------------------------";
  std::cerr << "\nFunction TESTING:\n" << std::endl;

  /** Just call all available public functions. */
  SODerivativeWeightFunctionType::IndexType startIndex;
  SODerivativeWeightFunctionType::IndexType trueStartIndex;
  trueStartIndex[0] = 2;
  trueStartIndex[1] = -4;
  soWeightFunction->ComputeStartIndex(cindex, startIndex);
  if (startIndex != trueStartIndex)
  {
    std::cerr << "ERROR: wrong start index was computed." << std::endl;
    return 1;
  }

  SODerivativeWeightFunctionType::SizeType trueSize;
  trueSize.Fill(SplineOrder + 1);
  if (soWeightFunction->GetSupportSize() != trueSize)
  {
    std::cerr << "ERROR: wrong support size was computed." << std::endl;
    return 1;
  }

  if (soWeightFunction->GetNumberOfWeights() !=
      static_cast<unsigned long>(std::pow(static_cast<float>(SplineOrder + 1), 2.0f)))
  {
    std::cerr << "ERROR: wrong number of weights was computed." << std::endl;
    return 1;
  }

  std::cerr << "All public functions returned valid output." << std::endl;

  /**
   * *********** PrintSelf TESTING ****************************************
   */

  std::cerr << "\n--------------------------------------------------------";
  std::cerr << "\nPrintSelf() TESTING:\n" << std::endl;

  soWeightFunction->Print(std::cerr, 0);

  /** Return a value. */
  return 0;

} // end main
