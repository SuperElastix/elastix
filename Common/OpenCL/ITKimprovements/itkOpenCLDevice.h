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
#ifndef itkOpenCLDevice_h
#define itkOpenCLDevice_h

#include "itkOpenCLPlatform.h"
#include "itkOpenCLSize.h"

namespace itk
{
/** \class OpenCLDevice
 * \brief The OpenCLDevice class represents the collection of OpenCL devices
 * to be used by the host.
 *
 * The OpenCL framework allows applications to use a host and one or more
 * OpenCL devices as a single heterogeneous parallel computer system.
 * The GetDeviceType() can be used to query specific OpenCL devices or all
 * OpenCL devices available. The valid values for GetDeviceType() are
 *
 * \table
 * \row \o Default \o The default OpenCL device in the system. The default
 * device cannot be a \c{CL_DEVICE_TYPE_CUSTOM} device.
 * \row \o CPU \o An OpenCL device that is the host processor. The host
 * processor runs the OpenCL implementations and is a single or multi-core CPU.
 * \row \o GPU \o An OpenCL device that is a GPU. By this we mean that the device
 * can also be used to accelerate a 3D API such as OpenGL or DirectX.
 * \row \o Accelerator \o Dedicated OpenCL accelerators (for example the IBM
 * CELL Blade). These devices communicate with the host processor using a
 * peripheral interconnect such as PCIe.
 * \row \o Custom \o Dedicated accelerators that do not support programs
 * written in OpenCL C.
 * \row \o All \o All OpenCL devices available in the system except
 * \c{CL_DEVICE_TYPE_CUSTOM} devices.
 * \endtable
 *
 * The GetDevices() function can be used to find all devices of a
 * specific type, optionally constrained by the OpenCLPlatform
 * they belong to.
 *
 * \ingroup OpenCL
 * \sa OpenCLPlatform, OpenCLContext
 */
class ITKOpenCL_EXPORT OpenCLDevice
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLDevice;

  /** Constructs a default OpenCL device identifier. */
  OpenCLDevice()
    : m_Id(0)
    , m_Version(0)
  {}

  /** Constructs an OpenCL device identifier that corresponds to the
   * native OpenCL value \a id. */
  OpenCLDevice(cl_device_id id)
    : m_Id(id)
    , m_Version(0)
  {}

  /** \enum OpenCLDevice::DeviceType
   * This enum defines the type of OpenCL device that is represented
   * by a OpenCLDevice object.
   * \value Default The default OpenCL device.
   * \value CPU The host CPU within the OpenCL system.
   * \value GPU An OpenCL device that is also an OpenGL GPU.
   * \value Accelerator Dedicated OpenCL accelerator.
   * \value All All OpenCL device types. */
  enum DeviceType
  {
    Default = (1 << 0),
    CPU = (1 << 1),
    GPU = (1 << 2),
    Accelerator = (1 << 3),
    Custom = (1 << 4),
    All = 0xFFFFFFFF
  };

  enum Endian
  {
    BigEndian,
    LittleEndian
  };

  /** Returns true if this OpenCL device identifier is null. */
  bool
  IsNull() const
  {
    return this->m_Id == 0;
  }

  /** Returns the native OpenCL device identifier for this object. */
  cl_device_id
  GetDeviceId() const
  {
    return this->m_Id;
  }

  /** Returns the OpenCL versions supported by this device.
   * \sa GetVersion(), OpenCLPlatform::GetOpenCLVersion() */
  OpenCLVersion
  GetOpenCLVersion() const;

  /** Returns the OpenCL version that is implemented by this OpenCL device,
   * usually something like \c{OpenCL 1.0}.
   * The versionFlags() function parses the version into flag bits
   * that are easier to test than the string returned by version().
   * \sa GetOpenCLVersion(), GetDriverVersion() */
  std::string
  GetVersion() const;

  /** Returns the type of this device.
   * It is possible for a device to have more than one type. */
  OpenCLDevice::DeviceType
  GetDeviceType() const;

  /** Returns the platform identifier for this device. */
  OpenCLPlatform
  GetPlatform() const;

  /** Returns the vendor's identifier for this device. */
  unsigned int
  GetVendorId() const;

  /** Returns true if this device is available, false otherwise. */
  bool
  IsAvailable() const;

  /** Returns true if this device has a compiler available, false otherwise. */
  bool
  HasCompiler() const;

  /** Returns true if this device has support for executing native kernels,
   * false otherwise. */
  bool
  HasNativeKernels() const;

  /** Returns true if this device supports out of order execution
   * of commands on a OpenCLCommandQueue, false otherwise.
   * \sa OpenCLCommandQueue::IsOutOfOrder() */
  bool
  HasOutOfOrderExecution() const;

  /** Returns true if this device supports the \c{double} type
   * via the \c{cl_khr_fp64} extension, false otherwise. */
  bool
  HasDouble() const;

  /** Returns true if this device supports operations on the
   * \c{half} type via the \c{cl_khr_fp16} extension, false otherwise.
   * Note: \c{half} is supported by the OpenCL 1.0 core specification
   * for data storage even if this function returns false.
   * However, kernels can only perform arithmetic operations on
   * \c{half} values if this function returns true. */
  bool
  HasHalfFloat() const;

  /** Returns true if the device implements error correction on
   * its memory areas, false otherwise. */
  bool
  HasErrorCorrectingMemory() const;

  /** Returns true if the device and the host share a unified
   * memory address space, false otherwise. */
  bool
  HasUnifiedMemory() const;

  /** Returns the number of parallel compute units on the device. */
  unsigned int
  GetComputeUnits() const;

  /** Returns the maximum clock frequency for this device in MHz. */
  unsigned int
  GetClockFrequency() const;

  /** Returns the number of address bits used by the device,
   * usually 32 or 64. */
  unsigned int
  GetAddressBits() const;

  /** Returns the byte order of the device, indicating little endian or big
   * endian. */
  OpenCLDevice::Endian
  GetByteOrder() const;

  /** Returns the maximum work size for this device.
   * \sa GetMaximumWorkItemsPerGroup() */
  OpenCLSize
  GetMaximumWorkItemSize() const;

  /** Returns the maximum number of work items in a work group executing a
   * kernel using data parallel execution.
   * \sa GetMaximumWorkItemSize() */
  std::size_t
  GetMaximumWorkItemsPerGroup() const;

  /** Returns true if this device has 2D image support, false otherwise. */
  bool
  HasImage2D() const;

  /** Returns true if this device has 3D image support, false otherwise. */
  bool
  HasImage3D() const;

  /** Returns true if this device supports writing to 3D images
   * via the \c{cl_khr_3d_image_writes} extension, false otherwise. */
  bool
  HasWritableImage3D() const;

  /** Returns the maximum size of 2D images that are supported
   * by this device; or an empty SizeType2D if images are not supported.
   * \sa GetMaximumImage3DSize(), HasImage2D() */
  OpenCLSize
  GetMaximumImage2DSize() const;

  /** Returns the maximum size of 3D images that are supported
   * by this device; or (0, 0, 0) if images are not supported.
   * \sa GetMaximumImage2DSize(), HasImage3D() */
  OpenCLSize
  GetMaximumImage3DSize() const;

  /** Returns the maximum number of image samplers that can be used
   * in a kernel at one time; 0 if images are not supported. */
  unsigned int
  GetMaximumSamplers() const;

  /** Returns the maximum number of image objects that can be
   * read simultaneously by a kernel; 0 if images are not supported.
   * \sa GetMaximumWriteImages() */
  unsigned int
  GetMaximumReadImages() const;

  /** Returns the maximum number of image objects that can be
   * written simultaneously by a kernel; 0 if images are not supported.
   * \sa maximumReadImages() */
  unsigned int
  GetMaximumWriteImages() const;

  /** Returns the preferred size for vectors of type \c{char} in the device.
   * For example, 4 indicates that 4 \c{char} values can be packed into
   * a vector and operated on as a unit for optimal performance. */
  unsigned int
  GetPreferredCharVectorSize() const;

  /** Returns the preferred size for vectors of type \c{short} in the device.
   * For example, 4 indicates that 4 \c{short} values can be packed into
   * a vector and operated on as a unit for optimal performance. */
  unsigned int
  GetPreferredShortVectorSize() const;

  /** Returns the preferred size for vectors of type \c{int} in the device.
   * For example, 4 indicates that 4 \c{int} values can be packed into
   * a vector and operated on as a unit for optimal performance. */
  unsigned int
  GetPreferredIntVectorSize() const;

  /** Returns the preferred size for vectors of type \c{long}
   * in the device. For example, 2 indicates that 2 \c{long}
   * values can be packed into a vector and operated on as a
   * unit for optimal performance. */
  unsigned int
  GetPreferredLongVectorSize() const;

  /** Returns the preferred size for vectors of type \c{float}
   * in the device. For example, 4 indicates that 4 \c{float}
   * values can be packed into a vector and operated on as a
   * unit for optimal performance. */
  unsigned int
  GetPreferredFloatVectorSize() const;

  /** Returns the preferred size for vectors of type \c{double}
   * in the device. For example, 2 indicates that 2 \c{double}
   * values can be packed into a vector and operated on as a
   * unit for optimal performance.
   * Returns zero if the device does not support \c{double}.
   * \sa HasDouble() */
  unsigned int
  GetPreferredDoubleVectorSize() const;

  /** Returns the preferred size for vectors of type \c{half}
   * in the device. For example, 2 indicates that 2 \c{half}
   * values can be packed into a vector and operated on as a
   * unit for optimal performance.
   * Returns zero if the device does not support \c{half},
   * or the device does not support OpenCL 1.1.
   * \sa HasHalfFloat() */
  unsigned int
  GetPreferredHalfFloatVectorSize() const;

  /** Returns the native size for vectors of type \c{char}
   * in the device. For example, 4 indicates that 4 \c{char}
   * values can be packed into a vector and operated on as a
   * unit for optimal performance. Returns zero on OpenCL 1.0. */
  unsigned int
  GetNativeCharVectorSize() const;

  /** Returns the native size for vectors of type \c{short}
   * in the device. For example, 4 indicates that 4 \c{short}
   * values can be packed into a vector and operated on as a
   * unit for optimal performance. Returns zero on OpenCL 1.0. */
  unsigned int
  GetNativeShortVectorSize() const;

  /** Returns the native size for vectors of type \c{int}
   * in the device. For example, 4 indicates that 4 \c{int}
   * values can be packed into a vector and operated on as a
   * unit for optimal performance. Returns zero on OpenCL 1.0. */
  unsigned int
  GetNativeIntVectorSize() const;

  /** Returns the native size for vectors of type \c{long}
   * in the device. For example, 2 indicates that 2 \c{long}
   * values can be packed into a vector and operated on as a
   * unit for optimal performance. Returns zero on OpenCL 1.0. */
  unsigned int
  GetNativeLongVectorSize() const;

  /** Returns the native size for vectors of type \c{float}
   * in the device. For example, 4 indicates that 4 \c{float}
   * values can be packed into a vector and operated on as a
   * unit for optimal performance. Returns zero on OpenCL 1.0. */
  unsigned int
  GetNativeFloatVectorSize() const;

  /** Returns the native size for vectors of type \c{double}
   * in the device. For example, 2 indicates that 2 \c{double}
   * values can be packed into a vector and operated on as a
   * unit for optimal performance. Returns zero on OpenCL 1.0,
   * or if the device does not support \c{double}.
   * \sa HasDouble() */
  unsigned int
  GetNativeDoubleVectorSize() const;

  /** Returns the native size for vectors of type \c{half}
   * in the device. For example, 2 indicates that 2 \c{half}
   * values can be packed into a vector and operated on as a
   * unit for optimal performance. Returns zero on OpenCL 1.0,
   * or if the device does not support \c{half}.
   * \sa HasHalfFloat() */
  unsigned int
  GetNativeHalfFloatVectorSize() const;

  /** \enum OpenCLDevice::FloatCapability
   * This enum defines the floating-point capabilities of the
   * \c{float} or \c{double} type on an OpenCL device.
   * \value NotSupported Returned to indicate that \c{double}
   * is not supported on the device.
   * \value Denorm Denorms are supported.
   * \value InfinityNaN Infinity and quiet NaN's are supported.
   * \value RoundNearest Round to nearest even rounding mode supported.
   * \value RoundZero Round to zero rounding mode supported.
   * \value RoundInfinity Round to infinity rounding mode supported.
   * \value FusedMultiplyAdd IEEE754-2008 fused multiply-add is supported. */
  enum FloatCapability
  {
    NotSupported = 0x0000,
    Denorm = 0x0001,
    InfinityNaN = 0x0002,
    RoundNearest = 0x0004,
    RoundZero = 0x0008,
    RoundInfinity = 0x0010,
    FusedMultiplyAdd = 0x0020
  };

  /** Returns a set of flags that describes the floating-point
   * capabilities of the \c{float} type on this device. */
  FloatCapability
  GetFloatCapabilities() const;

  /** Returns a set of flags that describes the floating-point
   * capabilities of the \c{double} type on this device.
   * Returns OpenCLDevice::NotSupported if operations on \c{double}
   * are not supported by the device.
   * \sa HasDouble() */
  FloatCapability
  GetDoubleCapabilities() const;

  /** Returns a set of flags that describes the floating-point
   * capabilities of the \c{half} type on this device.
   * Returns OpenCLDevice::NotSupported if operations on \c{half}
   * are not supported by the device.
   * \sa HasHalfFloat() */
  FloatCapability
  GetHalfFloatCapabilities() const;

  /** Returns the resolution of the device profiling timer in nanoseconds.
   * \sa OpenCLEvent::GetFinishTime() */
  std::size_t
  GetProfilingTimerResolution() const;

  /** Returns the maximum memory allocation size in bytes.
   * \sa GetGlobalMemorySize() */
  unsigned long
  GetMaximumAllocationSize() const;

  /** Returns the number of bytes of global memory in the device.
   * \sa GetGlobalMemoryCacheSize(), localMemorySize() */
  unsigned long
  GetGlobalMemorySize() const;

  /** \enum OpenCLDevice::CacheType
   * This enum defines the type of global memory cache that is
   * supported by an OpenCL device.
   * \value NoCache No global memory cache.
   * \value ReadOnlyCache Read-only global memory cache.
   * \value ReadWriteCache Read-write global memory cache. */
  enum CacheType
  {
    NoCache = 0,
    ReadOnlyCache = 1,
    ReadWriteCache = 2
  };

  /** Returns the type of global memory cache that is supported
   * by the device. */
  CacheType
  GetGlobalMemoryCacheType() const;

  /** Returns the size of the global memory cache in bytes.
   * \sa GetGlobalMemorySize(), GetGlobalMemoryCacheLineSize() */
  unsigned long
  GetGlobalMemoryCacheSize() const;

  /** Returns the size of a single global memory cache line in bytes.
   * \sa GetGlobalMemoryCacheSize() */
  unsigned int
  GetGlobalMemoryCacheLineSize() const;

  /** Returns the number of bytes of local memory in the device.
   * \sa GetGlobalMemorySize(), IsLocalMemorySeparate() */
  unsigned long
  GetLocalMemorySize() const;

  /** Returns true if the local memory on this device is in a separate
   * dedicated storage area from global memory; false if local memory
   * is allocated from global memory.
   * \sa GetLocalMemorySize() */
  bool
  IsLocalMemorySeparate() const;

  /** Returns the maximum size for a constant buffer allocation.
   * \sa GetMaximumConstantArguments() */
  unsigned long
  GetMaximumConstantBufferSize() const;

  /** Returns the maximum number of constant arguments that can
   * be passed to a kernel.
   * \sa GetMaximumConstantBufferSize() */
  unsigned int
  GetMaximumConstantArguments() const;

  /** Returns the default alignment for allocated memory in bytes.
   * \sa GetMinimumAlignment() */
  unsigned int
  GetDefaultAlignment() const;

  /** Returns the minimum alignment for any data type in bytes.
   * \sa GetDefaultAlignment() */
  unsigned int
  GetMinimumAlignment() const;

  /** Returns the maximum number of parameter bytes that can be passed
   * to a kernel. */
  std::size_t
  GetMaximumParameterBytes() const;

  /** Returns true if profile() is \c FULL_PROFILE, false otherwise.
   * \sa IsEmbeddedProfile() */
  bool
  IsFullProfile() const;

  /** Returns true if profile() is \c EMBEDDED_PROFILE, false otherwise.
   * \sa IsFullProfile() */
  bool
  IsEmbeddedProfile() const;

  /** Returns the profile that is implemented by this OpenCL device,
   * usually \c FULL_PROFILE or \c EMBEDDED_PROFILE.
   * \sa isFullProfile(), isEmbeddedProfile() */
  std::string
  GetProfile() const;

  /** Returns the driver version of this OpenCL device.
   * \sa GetVersion() */
  std::string
  GetDriverVersion() const;

  /** Returns the name of this OpenCL device. */
  std::string
  GetName() const;

  /** Returns the vendor of this OpenCL device. */
  std::string
  GetVendor() const;

  /** Returns a list of the extensions supported by this OpenCL device.
   * \sa HasExtension() */
  std::list<std::string>
  GetExtensions() const;

  /** Returns the highest version of the OpenCL language supported by
   * this device's compiler. For example, \c{OpenCL 1.1}. */
  std::string
  GetLanguageVersion() const;

  /** Returns true if this device has an extension called \a name, false otherwise.
   * This function is more efficient than checking for \a name in the return
   * value from extensions(), if the caller is only interested in a single
   * extension. Use extensions() to check for several extensions at once.
   * \sa GetExtensions() */
  bool
  HasExtension(const std::string & name) const;

  /** Returns a list of all OpenCL devices on all platforms on this system.
   * \sa GetDevices() */
  static std::list<OpenCLDevice>
  GetAllDevices();

  /** Returns a list of all OpenCL devices that match \a types on
   * \a platform on this system. If \a platform is null, then
   * the first platform that has devices matching \a types will be used.
   * \sa GetAllDevices() */
  static std::list<OpenCLDevice>
  GetDevices(const OpenCLDevice::DeviceType types, const OpenCLPlatform & platform = OpenCLPlatform());

  /** Returns a list of all OpenCL devices that match \a types on
   * \a vendor on this system.
   * \sa GetAllDevices() */
  static std::list<OpenCLDevice>
  GetDevices(const OpenCLDevice::DeviceType type, const OpenCLPlatform::VendorType vendor);

  /** Returns the device with maximal flops on this system that match \a type.
   * \sa GetAllDevices() */
  static OpenCLDevice
  GetMaximumFlopsDevice(const std::list<OpenCLDevice> & devices, const OpenCLDevice::DeviceType type);

  /** Returns the device with maximal flops on this system that match \a type.
   * \sa GetAllDevices(), GetMaximumFlopsDevices() */
  static OpenCLDevice
  GetMaximumFlopsDevice(const OpenCLDevice::DeviceType type);

  /** Returns the device with maximal flops on this system that match \a types and
   * \a vendor on this system.
   * \sa GetAllDevices(), GetMaximumFlopsDevices() */
  static OpenCLDevice
  GetMaximumFlopsDeviceByVendor(const OpenCLDevice::DeviceType type, const OpenCLPlatform::VendorType vendor);

  /** Returns the device with maximal flops from the context that match \a types on
   * \a platform on this system.
   * \sa GetMaximumFlopsDevice(), GetAllDevices() */
  static OpenCLDevice
  GetMaximumFlopsDeviceByPlatform(const OpenCLDevice::DeviceType types,
                                  const OpenCLPlatform &         platform = OpenCLPlatform());

  /** Returns the devices with maximal flops on this system, sorted according to
   * maximal flops. The device with maximal flops will be the first element
   * in the list container.
   * \sa GetAllDevices(), GetMaximumFlopsDevice() */
  static std::list<OpenCLDevice>
  GetMaximumFlopsDevices(const OpenCLDevice::DeviceType type, const OpenCLPlatform & platform = OpenCLPlatform());

private:
  cl_device_id m_Id;
  mutable int  m_Version;
};

/** Operator ==
 * Returns true if \a lhs OpenCL device identifier is the same as \a rhs, false otherwise.
 * \sa operator!= */
bool ITKOpenCL_EXPORT
     operator==(const OpenCLDevice & lhs, const OpenCLDevice & rhs);

/** Operator !=
 * Returns true if \a lhs OpenCL device identifier is not the same as \a rhs, false otherwise.
 * \sa operator== */
bool ITKOpenCL_EXPORT
     operator!=(const OpenCLDevice & lhs, const OpenCLDevice & rhs);

/** Stream out operator for OpenCLDevice */
template <typename charT, typename traits>
inline std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> & strm, const OpenCLDevice & device)
{
  if (device.IsNull())
  {
    strm << "OpenCLDevice(null)";
    return strm;
  }

  const char indent = ' ';

  strm << "OpenCLDevice\n" << indent << "Id: " << device.GetDeviceId() << std::endl;

  strm << indent << "OpenCL version: ";
  switch (device.GetOpenCLVersion())
  {
    case VERSION_1_0:
      strm << "1.0";
      break;
    case VERSION_1_1:
      strm << "1.1";
      break;
    case VERSION_1_2:
      strm << "1.2";
      break;
    case VERSION_2_0:
      strm << "2.0";
      break;
    case VERSION_2_1:
      strm << "2.1";
      break;
    default:
      strm << "Unknown";
      break;
  }

  strm << '\n' << indent << "Version: " << device.GetVersion() << std::endl;

  strm << indent << "Device type: ";
  switch (device.GetDeviceType())
  {
    case OpenCLDevice::Default:
      strm << "Default";
      break;
    case OpenCLDevice::CPU:
      strm << "CPU";
      break;
    case OpenCLDevice::GPU:
      strm << "GPU";
      break;
    case OpenCLDevice::Accelerator:
      strm << "Accelerator";
      break;
    case OpenCLDevice::All:
      strm << "All";
      break;
    default:
      strm << "Unknown";
      break;
  }

  // strm << device.GetPlatform() << std::endl;
  strm << '\n'
       << indent << "Vendor Id: " << device.GetVendorId() << '\n'
       << indent << "Available: " << (device.IsAvailable() ? "Yes" : "No") << '\n'
       << indent << "Has compiler: " << (device.HasCompiler() ? "Yes" : "No") << '\n'
       << indent << "Has native kernels: " << (device.HasNativeKernels() ? "Yes" : "No") << '\n'
       << indent << "Has out of order execution: " << (device.HasOutOfOrderExecution() ? "Yes" : "No") << '\n'
       << indent << "Has double: " << (device.HasDouble() ? "Yes" : "No") << '\n'
       << indent << "Has half float: " << (device.HasHalfFloat() ? "Yes" : "No") << '\n'
       << indent << "Has error correcting memory: " << (device.HasErrorCorrectingMemory() ? "Yes" : "No") << '\n'
       << indent << "Has unified memory: " << (device.HasUnifiedMemory() ? "Yes" : "No") << '\n'
       << indent << "Compute units: " << device.GetComputeUnits() << '\n'
       << indent << "Clock frequency: " << device.GetClockFrequency() << '\n'
       << indent << "Address bits: " << device.GetAddressBits() << std::endl;

  strm << indent << "Byte order: ";
  switch (device.GetByteOrder())
  {
    case OpenCLDevice::BigEndian:
      strm << "Big Endian";
      break;
    case OpenCLDevice::LittleEndian:
      strm << "Little Endian";
      break;
    default:
      strm << "Unknown";
      break;
  }

  strm << '\n'
       << indent << "Maximum work item size: " << device.GetMaximumWorkItemSize() << '\n'
       << indent << "Maximum work items per group: " << device.GetMaximumWorkItemsPerGroup() << '\n'
       << indent << "Maximum work items per group: " << device.GetMaximumWorkItemsPerGroup() << std::endl;

  // Has Image 2D
  const bool hasImage2D = device.HasImage2D();
  strm << indent << "Has image 2D: " << (hasImage2D ? "Yes" : "No") << std::endl;
  if (hasImage2D)
  {
    strm << indent << "Maximum image 2D size: " << device.GetMaximumImage2DSize() << std::endl;
  }

  // Has Image 3D
  const bool hasImage3D = device.HasImage3D();
  strm << indent << "Has image 3D: " << (hasImage3D ? "Yes" : "No") << std::endl;
  if (hasImage3D)
  {
    strm << indent << "Maximum image 3D size: " << device.GetMaximumImage3DSize() << std::endl;
  }

  strm << indent << "Has writable image 3D: " << (device.HasWritableImage3D() ? "Yes" : "No") << '\n'
       << indent << "Maximum samplers: " << device.GetMaximumSamplers() << '\n'
       << indent << "Maximum read images: " << device.GetMaximumReadImages() << '\n'
       << indent << "Maximum write images: " << device.GetMaximumWriteImages() << std::endl

       << indent << "Preferred char vector size: " << device.GetPreferredCharVectorSize() << '\n'
       << indent << "Preferred short vector size: " << device.GetPreferredShortVectorSize() << '\n'
       << indent << "Preferred int vector size: " << device.GetPreferredIntVectorSize() << '\n'
       << indent << "Preferred long vector size: " << device.GetPreferredLongVectorSize() << '\n'
       << indent << "Preferred float vector size: " << device.GetPreferredFloatVectorSize() << '\n'
       << indent << "Preferred double vector size: " << device.GetPreferredDoubleVectorSize() << '\n'
       << indent << "Preferred half float vector size: " << device.GetPreferredHalfFloatVectorSize() << std::endl

       << indent << "Native char vector size: " << device.GetNativeCharVectorSize() << '\n'
       << indent << "Native short vector size: " << device.GetNativeShortVectorSize() << '\n'
       << indent << "Native int vector size: " << device.GetNativeIntVectorSize() << '\n'
       << indent << "Native long vector size: " << device.GetNativeLongVectorSize() << '\n'
       << indent << "Native float vector size: " << device.GetNativeFloatVectorSize() << '\n'
       << indent << "Native double vector size: " << device.GetNativeDoubleVectorSize() << '\n'
       << indent << "Native half float vector size: " << device.GetNativeHalfFloatVectorSize() << std::endl;

  strm << indent << "Float capabilities: ";
  switch (device.GetFloatCapabilities())
  {
    case OpenCLDevice::NotSupported:
      strm << "Not supported";
      break;
    case OpenCLDevice::Denorm:
      strm << "Denorm";
      break;
    case OpenCLDevice::InfinityNaN:
      strm << "Infinity nan";
      break;
    case OpenCLDevice::RoundNearest:
      strm << "Round nearest";
      break;
    case OpenCLDevice::RoundZero:
      strm << "Round zero";
      break;
    case OpenCLDevice::RoundInfinity:
      strm << "Round infinity";
      break;
    case OpenCLDevice::FusedMultiplyAdd:
      strm << "Fused multiply add";
      break;
    default:
      strm << "Unknown";
      break;
  }

  strm << '\n' << indent << "Double capabilities: ";
  switch (device.GetDoubleCapabilities())
  {
    case OpenCLDevice::NotSupported:
      strm << "Not supported";
      break;
    case OpenCLDevice::Denorm:
      strm << "Denorm";
      break;
    case OpenCLDevice::InfinityNaN:
      strm << "Infinity NaN";
      break;
    case OpenCLDevice::RoundNearest:
      strm << "Round nearest";
      break;
    case OpenCLDevice::RoundZero:
      strm << "Round zero";
      break;
    case OpenCLDevice::RoundInfinity:
      strm << "Round infinity";
      break;
    case OpenCLDevice::FusedMultiplyAdd:
      strm << "Fused multiply add";
      break;
    default:
      strm << "Unknown";
      break;
  }

  strm << '\n' << indent << "Half float capabilities: ";
  switch (device.GetHalfFloatCapabilities())
  {
    case OpenCLDevice::NotSupported:
      strm << "Not supported";
      break;
    case OpenCLDevice::Denorm:
      strm << "Denorm";
      break;
    case OpenCLDevice::InfinityNaN:
      strm << "Infinity NaN";
      break;
    case OpenCLDevice::RoundNearest:
      strm << "Round nearest";
      break;
    case OpenCLDevice::RoundZero:
      strm << "Round zero";
      break;
    case OpenCLDevice::RoundInfinity:
      strm << "Round infinity";
      break;
    case OpenCLDevice::FusedMultiplyAdd:
      strm << "Fused multiply add";
      break;
    default:
      strm << "Unknown";
      break;
  }

  strm << '\n'
       << indent << "Profiling timer resolution: " << device.GetProfilingTimerResolution() << '\n'
       << indent << "Maximum allocation size: " << device.GetMaximumAllocationSize() << '\n'
       << indent << "Global memory size: " << device.GetGlobalMemorySize() << std::endl;

  strm << indent << "Global memory cache type: ";
  switch (device.GetGlobalMemoryCacheType())
  {
    case OpenCLDevice::NoCache:
      strm << "No cache";
      break;
    case OpenCLDevice::ReadOnlyCache:
      strm << "Read only cache";
      break;
    case OpenCLDevice::ReadWriteCache:
      strm << "Read write cache";
      break;
    default:
      strm << "Unknown";
      break;
  }

  strm << '\n'
       << indent << "Global memory cache size: " << device.GetGlobalMemoryCacheSize() << '\n'
       << indent << "Global memory cache line size: " << device.GetGlobalMemoryCacheLineSize() << '\n'
       << indent << "Local memory size: " << device.GetLocalMemorySize() << '\n'
       << indent << "Local memory separated: " << (device.IsLocalMemorySeparate() ? "Yes" : "No") << '\n'
       << indent << "Maximum constant buffer size: " << device.GetMaximumConstantBufferSize() << '\n'
       << indent << "Maximum constant arguments: " << device.GetMaximumConstantArguments() << '\n'
       << indent << "Default alignment: " << device.GetDefaultAlignment() << '\n'
       << indent << "Minimum alignment: " << device.GetMinimumAlignment() << '\n'
       << indent << "Maximum parameter bytes: " << device.GetMaximumParameterBytes() << '\n'
       << indent << "Full profile: " << (device.IsFullProfile() ? "Yes" : "No") << '\n'
       << indent << "Embedded profile: " << (device.IsEmbeddedProfile() ? "Yes" : "No") << '\n'
       << indent << "Profile: " << device.GetProfile() << '\n'
       << indent << "Driver version: " << device.GetDriverVersion() << '\n'
       << indent << "Name: " << device.GetName() << '\n'
       << indent << "Vendor: " << device.GetVendor() << '\n'
       << indent << "Language Version: " << device.GetLanguageVersion() << std::endl;

  const std::list<std::string> extensions = device.GetExtensions();
  const std::size_t            extensionsSize = extensions.size();
  strm << indent << "Extensions(" << extensionsSize << "): ";
  if (extensionsSize == 0)
  {
    strm << "none";
  }
  else
  {
    strm << std::endl;
    for (std::list<std::string>::const_iterator it = extensions.begin(); it != extensions.end(); ++it)
    {
      strm << indent << indent << "- " << *it << std::endl;
    }
  }

  return strm;
}


} // end namespace itk

#endif /* itkOpenCLDevice_h */
