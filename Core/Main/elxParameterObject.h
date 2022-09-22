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
#ifndef elxParameterObject_h
#define elxParameterObject_h

#include "itkObjectFactory.h"
#include "itkDataObject.h"
#include "elxMacro.h"

namespace elastix
{

// TODO: Why does the compiler not see ELASTIXLIB_API declspec in elxMacro.h?
//   error: variable has incomplete type 'class ELASTIXLIB_API'
// with class ELASTIXLIB_API ParameterObject : public itk::DataObject

class ParameterObject : public itk::DataObject
{
public:
  using Self = ParameterObject;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  itkNewMacro(Self);
  itkTypeMacro(ParameterObject, itk::DataObject);

  using ParameterKeyType = std::string;
  using ParameterValueType = std::string;
  using ParameterValueVectorType = std::vector<ParameterValueType>;
  using ParameterValueVectorIterator = ParameterValueVectorType::iterator;
  using ParameterMapType = std::map<ParameterKeyType, ParameterValueVectorType>;
  using ParameterMapIterator = ParameterMapType::iterator;
  using ParameterMapConstIterator = ParameterMapType::const_iterator;
  using ParameterMapVectorType = std::vector<ParameterMapType>;
  using ParameterFileNameType = std::string;
  using ParameterFileNameVectorType = std::vector<ParameterFileNameType>;
  using ParameterFileNameVectorIterator = ParameterFileNameVectorType::iterator;
  using ParameterFileNameVectorConstIterator = ParameterFileNameVectorType::const_iterator;

  /* Set/Get/Add parameter map or vector of parameter maps. */
  // TODO: Use itkSetMacro for ParameterMapVectorType
  void
  SetParameterMap(const ParameterMapType & parameterMap);
  void
  SetParameterMap(const unsigned int index, const ParameterMapType & parameterMap);
  void
  SetParameterMap(const ParameterMapVectorType & parameterMaps);
  void
  AddParameterMap(const ParameterMapType & parameterMap);
  const ParameterMapType &
  GetParameterMap(const unsigned int index) const;

  const ParameterMapVectorType &
  GetParameterMap() const
  {
    return m_ParameterMaps;
  }

  unsigned int
  GetNumberOfParameterMaps() const
  {
    return static_cast<unsigned int>(m_ParameterMaps.size());
  }

  void
  SetParameter(const unsigned int index, const ParameterKeyType & key, const ParameterValueType & value);
  void
  SetParameter(const unsigned int index, const ParameterKeyType & key, const ParameterValueVectorType & value);
  void
  SetParameter(const ParameterKeyType & key, const ParameterValueType & value);
  void
  SetParameter(const ParameterKeyType & key, const ParameterValueVectorType & value);
  const ParameterValueVectorType &
  GetParameter(const unsigned int index, const ParameterKeyType & key);
  void
  RemoveParameter(const unsigned int index, const ParameterKeyType & key);
  void
  RemoveParameter(const ParameterKeyType & key);

  /* Read/Write parameter file or multiple parameter files to/from disk. */
  void
  ReadParameterFile(const ParameterFileNameType & parameterFileName);
  void
  ReadParameterFile(const ParameterFileNameVectorType & parameterFileNameVector);
  void
  AddParameterFile(const ParameterFileNameType & parameterFileName);
  void
  WriteParameterFile() const;
  static void
  WriteParameterFile(const ParameterMapType & parameterMap, const ParameterFileNameType & parameterFileName);
  void
  WriteParameterFile(const ParameterFileNameType & parameterFileName) const;
  void
  WriteParameterFile(const ParameterFileNameVectorType & parameterFileNameVector) const;
  static void
  WriteParameterFile(const ParameterMapVectorType &      parameterMapVector,
                     const ParameterFileNameVectorType & parameterFileNameVector);

  /* Get preconfigured parameter maps. */
  static const ParameterMapType
  GetDefaultParameterMap(const std::string & transformName,
                         const unsigned int  numberOfResolutions = 4u,
                         const double        finalGridSpacingInPhysicalUnits = 10.0);

protected:
  void
  PrintSelf(std::ostream & os, itk::Indent indent) const override;

private:
  ParameterMapVectorType m_ParameterMaps;
};

} // namespace elastix

#endif // elxParameterObject_h
