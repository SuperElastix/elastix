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
#ifndef itkGPUCompositeTransformBase_h
#define itkGPUCompositeTransformBase_h

#include "itkGPUTransformBase.h"
#include "itkTransform.h"

namespace itk
{
/** \class GPUCompositeTransformBaseBase
 * \brief Base class for all GPU composite transforms.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TScalarType = float, unsigned int NDimensions = 3>
class ITK_TEMPLATE_EXPORT GPUCompositeTransformBase : public GPUTransformBase
{
public:
  /** Standard class typedefs. */
  typedef GPUCompositeTransformBase Self;
  typedef GPUTransformBase          GPUSuperclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUCompositeTransformBase, GPUSuperclass);

  /** Sub transform types. */
  typedef TScalarType                                      ScalarType;
  typedef Transform<TScalarType, NDimensions, NDimensions> TransformType;
  typedef typename TransformType::Pointer                  TransformTypePointer;
  typedef typename TransformType::ConstPointer             TransformTypeConstPointer;

  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, NDimensions);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, NDimensions);

  /** Get number of transforms in composite transform. */
  virtual SizeValueType
  GetNumberOfTransforms() const = 0;

  /** Get the Nth transform. */
  virtual const TransformTypePointer
  GetNthTransform(SizeValueType n) const = 0;

  /** Returns true if the derived composite transform has identity transform,
   * false otherwise. */
  virtual bool
  HasIdentityTransform(void) const;

  /** Returns true if the derived composite transform has matrix offset transform,
   * false otherwise. */
  virtual bool
  HasMatrixOffsetTransform(void) const;

  /** Returns true if the derived composite transform has translation transform,
   * false otherwise. */
  virtual bool
  HasTranslationTransform(void) const;

  /** Returns true if the derived composite transform has BSpline transform,
   * false otherwise. */
  virtual bool
  HasBSplineTransform(void) const;

  /** Returns true if the transform at \a index is identity transform,
   * false otherwise. */
  virtual bool
  IsIdentityTransform(const std::size_t index) const;

  /** Returns true if the transform at \a index is matrix offset transform,
   * false otherwise. */
  virtual bool
  IsMatrixOffsetTransform(const std::size_t index) const;

  /** Returns true if the transform at \a index is translation transform,
   * false otherwise. */
  virtual bool
  IsTranslationTransform(const std::size_t index) const;

  /** Returns true if the transform at \a index is BSpline transform,
   * false otherwise. */
  virtual bool
  IsBSplineTransform(const std::size_t index) const;

protected:
  GPUCompositeTransformBase() = default;
  ~GPUCompositeTransformBase() override = default;

  /** Returns OpenCL \a source code for the transform.
   * Returns true if source code was combined, false otherwise. */
  bool
  GetSourceCode(std::string & source) const override;

  /** Returns data manager that stores all settings for the transform \a index.
   * Used by combination transforms. */
  GPUDataManager::Pointer
  GetParametersDataManager(const std::size_t index) const override;

private:
  GPUCompositeTransformBase(const Self & other) = delete;
  const Self &
  operator=(const Self &) = delete;

  /** \internal
   * Returns true if the transform at \a index is identity transform,
   * false otherwise. If \a loadSource is true, the OpenCL \a source code is loaded. */
  bool
  IsIdentityTransform(const std::size_t index, const bool loadSource, std::string & source) const;

  /** \internal
   * Returns true if the transform at \a index is matrix offset transform,
   * false otherwise. If \a loadSource is true, the OpenCL \a source code is loaded. */

  bool
  IsMatrixOffsetTransform(const std::size_t index, const bool loadSource, std::string & source) const;

  /** \internal
   * Returns true if the transform at \a index is translation transform,
   * false otherwise. If \a loadSource is true, the OpenCL \a source code is loaded. */
  bool
  IsTranslationTransform(const std::size_t index, const bool loadSource, std::string & source) const;

  /** \internal
   * Returns true if the transform at \a index is BSpline transform,
   * false otherwise. If \a loadSource is true, the OpenCL \a source code is loaded. */
  bool
  IsBSplineTransform(const std::size_t index, const bool loadSource, std::string & source) const;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUCompositeTransformBase.hxx"
#endif

#endif /* itkGPUCompositeTransformBase_h */
