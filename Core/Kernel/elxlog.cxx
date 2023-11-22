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

// Its own header
#include "elxlog.h"

#include <cassert>
#include <fstream>
#include <iostream>
#ifndef __wasi__
#include <mutex>
#endif

namespace elastix
{
namespace
{

class logger
{
public:
  void
  set(const std::string & log_filename,
      const bool          do_log_to_file,
      const bool          do_log_to_stdout,
      const log::level    log_level)
  {
    m_data = { log_filename, do_log_to_file, do_log_to_stdout, log_level };
  }

  void
  reset()
  {
    m_data = {};
  }

  void
  to_file(const std::string & message)
  {
#ifndef __wasi__
    const std::lock_guard<std::mutex> lock(m_file_mutex);
#endif

    if (!m_data.log_filename.empty())
    {
      if (!m_data.file_stream.is_open())
      {
        m_data.file_stream.open(m_data.log_filename);
      }
      m_data.file_stream << message << std::endl;
    }
  }


  void
  to_stdout(const std::string & message)
  {
#ifndef __wasi__
    const std::lock_guard<std::mutex> lock(m_stdout_mutex);
#endif
    std::cout << message << std::endl;
  }

  void
  to_enabled_outputs(const std::string & message)
  {
    if (m_data.is_logging_to_stdout)
    {
      to_stdout(message);
    }
    if (m_data.is_logging_to_file)
    {
      to_file(message);
    }
  }

  auto
  get_log_level() const
  {
    return m_data.level;
  }

  auto
  is_logging_to_file() const
  {
    return m_data.is_logging_to_file;
  }

  auto
  is_logging_to_any_output() const
  {
    return m_data.is_logging_to_file || m_data.is_logging_to_stdout;
  }


private:
  // All data other than the mutexes.
  struct data
  {
    std::string   log_filename{};
    bool          is_logging_to_file{};
    bool          is_logging_to_stdout{};
    log::level    level{};
    std::ofstream file_stream{};
  };

  data m_data{};

#ifndef __wasi__
  std::mutex m_file_mutex{};
  std::mutex m_stdout_mutex{};
#endif
};


auto &
get_logger()
{
  static logger static_logger{};

  return static_logger;
}

std::string
get_string_from_stream(const std::ostream & stream)
{
  const auto * const outputstringstream = dynamic_cast<const std::ostringstream *>(&stream);

  if (outputstringstream)
  {
    return outputstringstream->str();
  }
  assert(!"Failed to convert std::ostream pointer to std::ostringstream pointer!");
  log::error("Failed to convert std::ostream pointer to std::ostringstream pointer");
  return {};
}


void
setup_implementation(const std::string & log_filename,
                     const bool          do_log_to_file,
                     const bool          do_log_to_stdout,
                     const log::level    log_level)
{
  get_logger().set(log_filename, do_log_to_file, do_log_to_stdout, log_level);
}

template <log::level log_level>
void
log_to_multi_logger(const std::string & message)
{
  auto & logger = get_logger();

  if (logger.is_logging_to_any_output() && (logger.get_log_level() <= log_level))
  {
    logger.to_enabled_outputs(message);
  }
}

template <log::level log_level>
void
log_to_multi_logger(const std::ostream & stream)
{
  auto & logger = get_logger();

  if (logger.is_logging_to_any_output() && (logger.get_log_level() <= log_level))
  {
    logger.to_enabled_outputs(get_string_from_stream(stream));
  }
}

} // namespace


bool
log::setup(const std::string & log_filename,
           const bool          do_log_to_file,
           const bool          do_log_to_stdout,
           const log::level    log_level)
{
  try
  {
    setup_implementation(log_filename, do_log_to_file, do_log_to_stdout, log_level);
    return true;
  }
  catch (const std::exception &)
  {
    return false;
  }
}


log::guard::guard() = default;

log::guard::guard(const std::string & log_filename,
                  const bool          do_log_to_file,
                  const bool          do_log_to_stdout,
                  const log::level    log_level)
{
  setup_implementation(log_filename, do_log_to_file, do_log_to_stdout, log_level);
}

log::guard::~guard()
{
  get_logger().reset();
}


void
log::info(const std::string & message)
{
  log_to_multi_logger<log::level::info>(message);
}

void
log::warn(const std::string & message)
{
  log_to_multi_logger<log::level::warn>(message);
}

void
log::error(const std::string & message)
{
  log_to_multi_logger<log::level::err>(message);
}


void
log::info(const std::ostream & stream)
{
  log_to_multi_logger<log::level::info>(stream);
}

void
log::warn(const std::ostream & stream)
{
  log_to_multi_logger<log::level::warn>(stream);
}

void
log::error(const std::ostream & stream)
{
  log_to_multi_logger<log::level::err>(stream);
}

void
log::info_to_log_file(const std::string & message)
{
  auto & logger = get_logger();

  if (logger.is_logging_to_file() && (logger.get_log_level() == level::info))
  {
    logger.to_file(message);
  }
}

void
log::info_to_log_file(const std::ostream & stream)
{
  auto & logger = get_logger();

  if (logger.is_logging_to_file() && (logger.get_log_level() == level::info))
  {
    logger.to_file(get_string_from_stream(stream));
  }
}

void
log::to_stdout(const std::string & message)
{
  get_logger().to_stdout(message);
}

void
log::to_stdout(const std::ostream & stream)
{
  get_logger().to_stdout(get_string_from_stream(stream));
}

} // end namespace elastix
