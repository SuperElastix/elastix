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
#ifndef itkBSplineStackTransform_h
#define itkBSplineStackTransform_h

#include "itkStackTransform.h"
#include "itkAdvancedBSplineDeformableTransform.h"
#include "elxElastixBase.h"

namespace itk
{
template <unsigned int NDimension>
class ITK_TEMPLATE_EXPORT BSplineStackTransform
  : public itk::StackTransform<elx::ElastixBase::CoordRepType, NDimension, NDimension>
{
private:
  using CoordRepType = elx::ElastixBase::CoordRepType;

public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineStackTransform);

  using Self = BSplineStackTransform;
  using Superclass = itk::StackTransform<CoordRepType, NDimension, NDimension>;
  using Pointer = itk::SmartPointer<BSplineStackTransform>;
  using typename Superclass::FixedParametersType;
  itkNewMacro(Self);
  itkTypeMacro(BSplineStackTransform, Superclass);

private:
  using Superclass::NumberOfGeneralFixedParametersOfStack;

  static constexpr unsigned int NumberOfFixedParametersOfSubTransform =
    AdvancedBSplineDeformableTransformBase<CoordRepType, NDimension - 1>::NumberOfFixedParameters;

  static constexpr unsigned int NumberOfFixedParameters =
    NumberOfGeneralFixedParametersOfStack + NumberOfFixedParametersOfSubTransform + 1;

public:
  void
  SetSplineOrder(const unsigned newValue)
  {
    m_SplineOrder = newValue;

    if (!Superclass::m_FixedParameters.empty())
    {
      Superclass::m_FixedParameters.back() = m_SplineOrder;
    }
  }

protected:
  /** Default-constructor */
  BSplineStackTransform() = default;

  /** Destructor */
  ~BSplineStackTransform() override = default;

  void
  SetFixedParameters(const FixedParametersType & fixedParameters) override
  {
    const auto numberOfFixedParameters = fixedParameters.size();
    if (numberOfFixedParameters < NumberOfFixedParameters)
    {
      itkExceptionMacro(<< "The number of FixedParameters (" << numberOfFixedParameters << ") should be at least "
                        << NumberOfFixedParameters);
    }
    const auto lastFixedParameter = fixedParameters.back();
    if (lastFixedParameter >= 1 && lastFixedParameter <= 3 &&
        static_cast<double>(static_cast<unsigned>(lastFixedParameter)) == lastFixedParameter)
    {
      m_SplineOrder = static_cast<unsigned>(fixedParameters.back());
    }
    else
    {
      itkExceptionMacro(<< "The last FixedParameters (" << lastFixedParameter << ") should be a valid spline order.");
    }

    if (Superclass::m_FixedParameters != fixedParameters)
    {
      Superclass::m_FixedParameters = fixedParameters;

      Superclass::CreateSubTransforms(FixedParametersType(
        fixedParameters.data_block() + NumberOfGeneralFixedParametersOfStack, NumberOfFixedParametersOfSubTransform));
      Superclass::UpdateStackSpacingAndOrigin();
      this->Modified();
    }
  }

private:
  void
  UpdateFixedParametersInternally(const FixedParametersType & fixedParametersOfSubTransform) override
  {
    FixedParametersType & fixedParametersOfStack = this->Superclass::m_FixedParameters;
    fixedParametersOfStack.set_size(NumberOfGeneralFixedParametersOfStack + NumberOfFixedParametersOfSubTransform + 1);
    fixedParametersOfStack.back() = m_SplineOrder;
    Superclass::UpdateFixedParametersInternally(fixedParametersOfSubTransform);
  }

  /** Create a subtransform that may be added to this specific stack. */
  typename Superclass::SubTransformPointer
  CreateSubTransform() const override
  {
    return AdvancedBSplineDeformableTransformBase<CoordRepType, NDimension - 1>::template Create<
      AdvancedBSplineDeformableTransform>(m_SplineOrder);
  }

  unsigned m_SplineOrder{};
};

} // namespace itk

#endif
