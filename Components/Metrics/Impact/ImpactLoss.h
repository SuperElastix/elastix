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

/**
 * \file ImpactLoss.h
 *
 * Implementation of differentiable loss functions used in the IMPACT image registration metric.
 * Each loss is implemented as a class inheriting from `Loss`, and can be registered dynamically
 * via the `LossFactory` for use at runtime.
 *
 * Supports both static and Jacobian-based backpropagation modes.
 *
 * \author V. Boussot,  Univ. Rennes, INSERM, LTSI- UMR 1099, F-35000 Rennes, France
 * \note This work was funded by the French National Research Agency as part of the VATSop project (ANR-20-CE19-0015).
 * \note If you use the Impact anywhere we would appreciate if you cite the following article:\n
 * V. Boussot et al., IMPACT: A Generic Semantic Loss for Multimodal Medical Image Registration, arXiv preprint
 * arXiv:2503.24121 (2025). https://doi.org/10.48550/arXiv.2503.24121
 *
 */

#ifndef _ImpactLoss_h
#define _ImpactLoss_h

#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include "itkTimeProbe.h"

namespace ImpactLoss
{

/**
 * \class Loss
 * \brief Abstract base class for losses operating on extracted feature maps.
 *
 * Stores the accumulated loss value and its derivative, and provides methods for updating them.
 * Designed to support both:
 *   - Static mode (with manual update of derivative using precomputed jacobians)
 *   - Jacobian mode (direct backpropagation of gradients)
 *
 * Subclasses must implement:
 *   - updateValue()
 *   - updateValueAndGetGradientModulator()
 */
class Loss
{
private:
  mutable double m_Normalization = 0;

protected:
  double        m_Value;
  torch::Tensor m_Derivative;
  bool          m_Initialized = false;
  int           m_NumberOfParameters;

public:
  Loss(bool isLossNormalized)
  {
    if (!isLossNormalized)
    {
      m_Normalization = 1.0;
    }
  }

  void
  setNumberOfParameters(int numberOfParameters)
  {
    m_NumberOfParameters = numberOfParameters;
  }
  void
  reset()
  {
    m_Initialized = false;
  }

  virtual void
  initialize(torch::Tensor & output)
  {
    // Lazy initialization of internal buffers based on output tensor shape and number of parameters
    if (!m_Initialized)
    {
      m_Value = 0;
      m_Derivative = torch::zeros({ m_NumberOfParameters }, output.options());
      m_Initialized = true;
    }
  }

  virtual void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) = 0;
  virtual void
  updateValueAndDerivativeInStaticMode(torch::Tensor & fixedOutput,
                                       torch::Tensor & movingOutput,
                                       torch::Tensor & jacobian,
                                       torch::Tensor & nonZeroJacobianIndices)
  {
    m_Derivative.index_add_(
      0,
      nonZeroJacobianIndices.flatten(),
      (updateValueAndGetGradientModulator(fixedOutput, movingOutput).unsqueeze(-1) * jacobian).sum(1).flatten());
  }
  virtual torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) = 0;
  void
  updateDerivativeInJacobianMode(torch::Tensor & jacobian, torch::Tensor & nonZeroJacobianIndices)
  {
    m_Derivative.index_add_(0, nonZeroJacobianIndices.flatten(), jacobian.flatten());
  }

  virtual double
  GetValue(double N) const
  {
    if (m_Normalization == 0)
    {
      m_Normalization = 1 / (m_Value / N);
    }
    return m_Normalization * m_Value / N;
  }

  virtual torch::Tensor
  GetDerivative(double N) const
  {
    return m_Normalization * m_Derivative.to(torch::kCPU) / N;
  }

  virtual ~Loss() = default;

  virtual Loss &
  operator+=(const Loss & other)
  {
    if (!m_Initialized && other.m_Initialized)
    {
      m_Value = other.m_Value;
      m_Derivative = other.m_Derivative;
      m_Initialized = true;
    }
    else if (other.m_Initialized)
    {
      m_Value += other.m_Value;
      m_Derivative += other.m_Derivative;
    }
    return *this;
  }
};

/**
 * \class LossFactory
 * \brief Singleton factory to register and create Loss instances by string name.
 *
 * Used to instantiate losses dynamically from configuration.
 * Example: "L1", "L2", "NCC", etc.
 */
class LossFactory
{
public:
  using CreatorFunc = std::function<std::unique_ptr<Loss>()>;

  static LossFactory &
  Instance()
  {
    static LossFactory instance;
    return instance;
  }

  void
  RegisterLoss(const std::string & name, CreatorFunc creator)
  {
    factoryMap[name] = creator;
  }

  std::unique_ptr<Loss>
  Create(const std::string & name)
  {
    auto it = factoryMap.find(name);
    if (it != factoryMap.end())
    {
      return it->second();
    }
    throw std::runtime_error("Error: Unknown loss function " + name);
  }

private:
  std::unordered_map<std::string, CreatorFunc> factoryMap;
};

template <typename T>
class RegisterLoss
{
public:
  RegisterLoss(const std::string & name)
  {
    LossFactory::Instance().RegisterLoss(name, []() { return std::make_unique<T>(); });
  }
};

/**
 * \class L1
 * \brief L1 loss over feature vectors: mean absolute difference.
 */
class L1 : public Loss
{
public:
  L1()
    : Loss(true)
  {}

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    this->m_Value += (fixedOutput - movingOutput).abs().mean(1).sum().item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor diffOutput = fixedOutput - movingOutput;
    this->m_Value += diffOutput.abs().mean(1).sum().item<double>();
    return -torch::sign(diffOutput) / fixedOutput.size(1);
  }
};

inline RegisterLoss<L1> L1_reg("L1"); // Register the loss under its string name for factory-based creation

/**
 * \class L2
 * \brief Mean Squared Error (L2) over feature vectors.
 */
class L2 : public Loss
{
public:
  L2()
    : Loss(true)
  {}

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    this->m_Value += (fixedOutput - movingOutput).pow(2).mean(1).sum().item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor diffOutput = fixedOutput - movingOutput;
    this->m_Value += diffOutput.pow(2).mean(1).sum().item<double>();
    return -2 * diffOutput / fixedOutput.size(1);
  }
};


inline RegisterLoss<L2> MSE_reg("L2"); // Register the loss under its string name for factory-based creation

/**
 * \class Dice
 * \brief Binary Dice loss (assumes thresholded activations).
 *
 * Rounds inputs to {0, 1} before computing overlap.
 */
class Dice : public Loss
{
public:
  Dice()
    : Loss(false)
  {}

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    fixedOutput = torch::round(fixedOutput).clamp(0);
    movingOutput = torch::round(movingOutput).clamp(0);

    torch::Tensor intersection = (fixedOutput * movingOutput).sum(1);
    torch::Tensor unionSum = (fixedOutput + movingOutput).sum(1);
    torch::Tensor diceScore = (2 * intersection + 1e-6) / (unionSum + 1e-6);

    this->m_Value += (1 - diceScore.mean()).item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    fixedOutput = torch::round(fixedOutput).clamp(0);
    movingOutput = torch::round(movingOutput).clamp(0);
    torch::Tensor intersection = (fixedOutput * movingOutput).sum(1);
    torch::Tensor unionSum = (fixedOutput + movingOutput).sum(1);
    torch::Tensor diceScore = (2 * intersection + 1e-6) / (unionSum + 1e-6);

    this->m_Value += (1 - diceScore.mean()).item<double>();

    return -(2 * (fixedOutput * unionSum.unsqueeze(-1) - intersection.unsqueeze(-1)) /
             (unionSum * unionSum + 1e-6).unsqueeze(-1));
  }
};


inline RegisterLoss<Dice> Dice_reg("Dice"); // Register the loss under its string name for factory-based creation

/**
 * \class L1Cosine
 * \brief Combined cosine similarity and exponential L1 loss.
 *
 * Useful for simultaneously penalizing direction and magnitude.
 */
class L1Cosine : public Loss
{
private:
  double m_Lambda;

public:
  L1Cosine()
    : Loss(false)
  {
    m_Lambda = 0.1;
  }

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor dotProduct = (fixedOutput * movingOutput).sum(1);
    torch::Tensor normFixed = torch::norm(fixedOutput, 2, 1);
    torch::Tensor normMoving = torch::norm(movingOutput, 2, 1);
    torch::Tensor cosine = dotProduct / (normFixed * normMoving);
    torch::Tensor expL1 = torch::exp(-m_Lambda * (fixedOutput - movingOutput).abs());
    this->m_Value -= (cosine.unsqueeze(-1) * expL1).mean(1).sum().item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor diffOutput = fixedOutput - movingOutput;
    torch::Tensor dotProduct = (fixedOutput * movingOutput).sum(1);
    torch::Tensor normFixed = torch::norm(fixedOutput, 2, 1);
    torch::Tensor normMoving = torch::norm(movingOutput, 2, 1);
    torch::Tensor v = (normFixed * normMoving);

    torch::Tensor cosine = dotProduct / (v);
    torch::Tensor expL1 = torch::exp(-m_Lambda * (fixedOutput - movingOutput).abs());

    torch::Tensor dCosine = -(fixedOutput / v.unsqueeze(-1) -
                              (fixedOutput * movingOutput * movingOutput) / (v * normMoving.pow(2)).unsqueeze(-1));
    torch::Tensor dexpL1 = -torch::sign(diffOutput) * expL1 / fixedOutput.size(1);
    this->m_Value -= (cosine.unsqueeze(-1) * expL1).mean(1).sum().item<double>();
    return dCosine * dexpL1 + cosine.unsqueeze(-1) * dexpL1;
  }
};

inline RegisterLoss<L1Cosine> L1CosineReg(
  "L1Cosine"); // Register the loss under its string name for factory-based creation

/**
 * \class Cosine
 * \brief Cosine similarity loss (negative mean cosine between vectors).
 */
class Cosine : public Loss
{
public:
  Cosine()
    : Loss(false)
  {}

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor dotProduct = (fixedOutput * movingOutput).sum(1);
    torch::Tensor normFixed = torch::norm(fixedOutput, 2, 1);
    torch::Tensor normMoving = torch::norm(movingOutput, 2, 1);
    this->m_Value -= (dotProduct / (normFixed * normMoving)).sum().item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    torch::Tensor dotProduct = (fixedOutput * movingOutput).sum(1);
    torch::Tensor normFixed = torch::norm(fixedOutput, 2, 1);
    torch::Tensor normMoving = torch::norm(movingOutput, 2, 1);
    torch::Tensor v = (normFixed * normMoving);
    this->m_Value -= (dotProduct / v).sum().item<double>();
    return -(fixedOutput / v.unsqueeze(-1) -
             (fixedOutput * movingOutput * movingOutput) / (v * normMoving.pow(2)).unsqueeze(-1));
  }
};

inline RegisterLoss<Cosine> CosineReg("Cosine"); // Register the loss under its string name for factory-based creation

/**
 * \class DotProduct
 * \brief Negative dot product loss (simple similarity).
 */
class DotProduct : public Loss
{
public:
  DotProduct()
    : Loss(false)
  {}

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    this->m_Value -= (fixedOutput * movingOutput).sum(1).sum().item<double>();
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    this->m_Value -= (fixedOutput * movingOutput).sum(1).sum().item<double>();
    return -fixedOutput;
  }
};


inline RegisterLoss<DotProduct> DotProductReg(
  "DotProduct"); // Register the loss under its string name for factory-based creation

/**
 * \class NCC
 * \brief Normalized Cross Correlation loss over feature vectors.
 *
 * Computes NCC between fixed and moving features across batches.
 * Derivative is accumulated in static mode using full Jacobian tracking.
 */
class NCC : public Loss
{
private:
  torch::Tensor m_Sff, m_Smm, m_Sfm, m_Sf, m_Sm;
  torch::Tensor m_Sfdm, m_Smdm, m_Sdm;

public:
  NCC()
    : Loss(false)
  {}

  void
  initialize(torch::Tensor & output) override
  {
    if (!this->m_Initialized)
    {
      m_Sff = torch::zeros({ output.size(1) }, output.options());
      m_Smm = torch::zeros({ output.size(1) }, output.options());
      m_Sfm = torch::zeros({ output.size(1) }, output.options());
      m_Sf = torch::zeros({ output.size(1) }, output.options());
      m_Sm = torch::zeros({ output.size(1) }, output.options());
      m_Initialized = true;
    }
  }

  void
  updateValue(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    this->initialize(fixedOutput);
    m_Sff += (fixedOutput * fixedOutput).sum(0);
    m_Smm += (movingOutput * movingOutput).sum(0);
    m_Sfm += (fixedOutput * movingOutput).sum(0);
    m_Sf += fixedOutput.sum(0);
    m_Sm += movingOutput.sum(0);
  }

  void
  updateValueAndDerivativeInStaticMode(torch::Tensor & fixedOutput,
                                       torch::Tensor & movingOutput,
                                       torch::Tensor & jacobian,
                                       torch::Tensor & nonZeroJacobianIndices) override
  {
    // Accumulate first-order statistics and weighted Jacobians
    // sfdm: sum(fixed * dM), smdm: sum(moving * dM), sdm: sum(dM)
    if (!this->m_Initialized)
    {
      m_Sfdm = torch::zeros({ fixedOutput.size(1), m_NumberOfParameters }, fixedOutput.options());
      m_Smdm = torch::zeros({ fixedOutput.size(1), m_NumberOfParameters }, fixedOutput.options());
      m_Sdm = torch::zeros({ fixedOutput.size(1), m_NumberOfParameters }, fixedOutput.options());
    }
    this->updateValue(fixedOutput, movingOutput);
    m_Sfdm.index_add_(
      1, nonZeroJacobianIndices.flatten(), (fixedOutput.unsqueeze(-1) * jacobian).permute({ 1, 0, 2 }).flatten(1, 2));
    m_Smdm.index_add_(
      1, nonZeroJacobianIndices.flatten(), (movingOutput.unsqueeze(-1) * jacobian).permute({ 1, 0, 2 }).flatten(1, 2));
    m_Sdm.index_add_(1, nonZeroJacobianIndices.flatten(), (jacobian).permute({ 1, 0, 2 }).flatten(1, 2));
  }

  torch::Tensor
  updateValueAndGetGradientModulator(torch::Tensor & fixedOutput, torch::Tensor & movingOutput) override
  {
    if (!this->m_Initialized)
    {
      this->m_Derivative = torch::zeros({ this->m_NumberOfParameters }, fixedOutput.options());
    }
    this->initialize(fixedOutput);

    const double  N = fixedOutput.size(0);
    torch::Tensor sff = (fixedOutput * fixedOutput).sum(0);
    torch::Tensor smm = (movingOutput * movingOutput).sum(0);
    torch::Tensor sfm = (fixedOutput * movingOutput).sum(0);
    torch::Tensor sf = fixedOutput.sum(0);
    torch::Tensor sm = movingOutput.sum(0);

    m_Sff += sff;
    m_Smm += smm;
    m_Sfm += sfm;
    m_Sf += sf;
    m_Sm += sm;

    torch::Tensor u = sfm - (sf * sm / N);
    torch::Tensor v = torch::sqrt(sff - sf * sf / N) * torch::sqrt(smm - sm * sm / N); // v = a*b

    torch::Tensor u_p = fixedOutput - sf.unsqueeze(0) / N;
    return -((u_p - u.unsqueeze(0) * (movingOutput - sm.unsqueeze(0) / N) / (smm - sm * sm / N).unsqueeze(0)) /
             v.unsqueeze(0)) /
           fixedOutput.size(1);
  }

  double
  GetValue(double N) const override
  {
    // Compute NCC loss from accumulated statistics: mean( -NCC(channel) )
    if (N <= 0)
      return 0.0;
    torch::Tensor u = m_Sfm - (m_Sf * m_Sm / N);
    torch::Tensor v = torch::sqrt(m_Sff - m_Sf * m_Sf / N) * torch::sqrt(m_Smm - m_Sm * m_Sm / N);
    return -(u / v).mean().item<double>();
  }

  torch::Tensor
  GetDerivative(double N) const override
  {
    if (this->m_Derivative.defined())
    {
      return this->m_Derivative.to(torch::kCPU);
    }

    torch::Tensor u = m_Sfm - (m_Sf * m_Sm / N);
    torch::Tensor v = torch::sqrt(m_Sff - m_Sf * m_Sf / N) * torch::sqrt(m_Smm - m_Sm * m_Sm / N);
    torch::Tensor u_p = m_Sfdm - m_Sf.unsqueeze(-1) * m_Sdm / N;
    return -((u_p -
              u.unsqueeze(-1) * (m_Smdm - m_Sm.unsqueeze(-1) * m_Sdm / N) / (m_Smm - m_Sm * m_Sm / N).unsqueeze(-1)) /
             v.unsqueeze(-1))
              .mean(0)
              .to(torch::kCPU);
  }

  NCC &
  operator+=(const Loss & other) override
  {
    const auto * nccOther = dynamic_cast<const NCC *>(&other);
    if (nccOther)
    {
      m_Sff += nccOther->m_Sff;
      m_Smm += nccOther->m_Smm;
      m_Sfm += nccOther->m_Sfm;
      m_Sf += nccOther->m_Sf;
      m_Sm += nccOther->m_Sm;
      if (m_Sfdm.defined())
      {
        m_Sfdm += nccOther->m_Sfdm;
        m_Smdm += nccOther->m_Smdm;
        m_Sdm += nccOther->m_Sdm;
      }
      if (m_Derivative.defined())
      {
        m_Derivative += nccOther->m_Derivative;
      }
    }
    return *this;
  }
};

inline RegisterLoss<NCC> NCC_reg("NCC"); // Register the loss under its string name for factory-based creation

} // namespace ImpactLoss

#endif // _ImpactLoss_h
