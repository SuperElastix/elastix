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

#ifndef itkScaledSingleValuedCostFunction_h
#define itkScaledSingleValuedCostFunction_h

#include "itkSingleValuedCostFunction.h"
#include "itkIntTypes.h" //temp, needed for IdentifierType

namespace itk
{
/**
 * \class ScaledSingleValuedCostFunction
 * \brief A cost function that applies a scaling to another cost function
 *
 * This class can be used to adapt an existing, badly scaled, cost function.
 *
 * By default it does not apply any scaling. Use the method SetUseScales(true)
 * to enable the use of scales.
 *
 * \ingroup Numerics
 */

class ScaledSingleValuedCostFunction : public SingleValuedCostFunction
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ScaledSingleValuedCostFunction);

  /** Standard ITK-stuff. */
  using Self = ScaledSingleValuedCostFunction;
  using Superclass = SingleValuedCostFunction;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ScaledSingleValuedCostFunction, SingleValuedCostFunction);

  /** Typedefs inherited from the superclass. */
  using Superclass::MeasureType;
  using Superclass::DerivativeType;
  using Superclass::ParametersType;
  // temporary, untill it is fixed in the ITK4
  // typedef IdentifierType NumberOfParametersType; // temp, copied from itk::TransformBase
  using NumberOfParametersType = unsigned int; // temp, copied from itk::CostFunction
  using SingleValuedCostFunctionPointer = Superclass::Pointer;

  using ScalesType = Array<double>;

  /** Divide the parameters by the scales and call the GetValue routine
   * of the unscaled cost function.
   */
  MeasureType
  GetValue(const ParametersType & parameters) const override;

  /** Divide the parameters by the scales, call the GetDerivative routine
   * of the unscaled cost function and divide the resulting derivative by
   * the scales.
   */
  void
  GetDerivative(const ParametersType & parameters, DerivativeType & derivative) const override;

  /** Same procedure as in GetValue and GetDerivative. */
  void
  GetValueAndDerivative(const ParametersType & parameters,
                        MeasureType &          value,
                        DerivativeType &       derivative) const override;

  /** Ask the UnscaledCostFunction how many parameters it has. */
  NumberOfParametersType
  GetNumberOfParameters() const override;

  /** Set the cost function that needs scaling. */
  itkSetObjectMacro(UnscaledCostFunction, Superclass);
  /** Get the cost function that needs scaling. */
  itkGetModifiableObjectMacro(UnscaledCostFunction, Superclass);

  /** Set the scales. Also computes the squared scales, just in case users
   * call GetSquaredScales (for compatibility with the ITK convention). */
  virtual void
  SetScales(const ScalesType & scales);

  /** Get the scales. */
  itkGetConstReferenceMacro(Scales, ScalesType);

  /** The ITK convention is to use the squared scales. This function
   * takes the square root of the input scales and sets them as the
   * the actual scales */
  virtual void
  SetSquaredScales(const ScalesType & squaredScales);

  /** The ITK convention is to use the squared scales. This function
   * returns the squared actual scales. */
  itkGetConstReferenceMacro(SquaredScales, ScalesType);

  /** Set the flag to use scales or not. */
  itkSetMacro(UseScales, bool);

  /** Get the flag to use scales or not. */
  itkGetConstMacro(UseScales, bool);

  /** Set the flag to negate the cost function or not. */
  itkBooleanMacro(NegateCostFunction);

  /** Set the flag to negate the cost function or not. */
  itkSetMacro(NegateCostFunction, bool);
  /** Get the flag to negate the cost function or not. */
  itkGetConstMacro(NegateCostFunction, bool);

  /** Convert the parameters from scaled to unscaled: x = y/s. */
  virtual void
  ConvertScaledToUnscaledParameters(ParametersType & parameters) const;

  /** Convert the parameters from unscaled to scaled: y = x*s. */
  virtual void
  ConvertUnscaledToScaledParameters(ParametersType & parameters) const;

protected:
  /** The constructor. */
  ScaledSingleValuedCostFunction();
  /** The destructor. */
  ~ScaledSingleValuedCostFunction() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  /** Member variables. */
  ScalesType                      m_Scales;
  ScalesType                      m_SquaredScales;
  SingleValuedCostFunctionPointer m_UnscaledCostFunction;
  bool                            m_UseScales;
  bool                            m_NegateCostFunction;
};

} // end namespace itk

#endif // #ifndef itkScaledSingleValuedCostFunction_h
